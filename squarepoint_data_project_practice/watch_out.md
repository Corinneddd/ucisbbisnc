**Dataset interview：数据处理、EDA 与预测建模检查清单 / Data processing, EDA and predictive modeling checklist**

适用范围：以表格预测任务为主，单独标出时间序列事项。清单用于决定下一步，而不是要求把所有方法都运行一遍。优先完成一个可验证的 baseline，再按诊断结果改进。

Scope: tabular prediction tasks, with specific reminders for time series. This checklist guides decisions; it does not require running every technique. Build and validate a baseline early, then improve it based on evidence.

**中文版：现场按这个顺序做**

**1. 先明确问题和预测时可用的信息**

- [ ] 一行代表什么：一个用户、一笔交易、一个设备，还是一个小时？
- [ ] Response / target 是什么？是回归、分类还是时间序列预测？单位是什么？
- [ ] 预测用于什么决策？预测错误的成本是否对称？
- [ ] 什么时候发出预测？预测多远？当时哪些 predictor 已经可获得？注意数据发布延迟。
- [ ] 确定主要评价指标及 baseline；如果目标由自己选择，解释业务意义和可预测性。

可以说：“我先明确预测对象、预测时点和评价标准，这样才能判断哪些特征可以合法使用。”

时间序列提醒：一次预测明天完整的 24 小时，与每小时更新后预测下一小时，是两个不同任务。前者不能在预测后半天时使用明天上午的真实数据。

**2. 读取数据，弄清字段含义**

- [ ] 查看 shape、head、dtypes、数值统计、类别频数与唯一值数量。
- [ ] 查数据字典、单位、正负号含义、ID 与名称映射；保留原始字段和 mapping。
- [ ] 检查主键；判断重复行是录入错误、重复采样，还是合理的多条记录。
- [ ] 区分 ID、数值、类别、时间、标签，以及由其他字段计算出来的字段。
- [ ] 如果合并多张表，检查 join key 的唯一性、关系是一对一还是一对多，以及合并前后行数。

应产出：字段说明表，以及每个字段在预测时是否可用。

**3. 检查数据质量和时间结构**

- [ ] 检查缺失值、无穷值、解析失败，以及用 -999、空字符串等表示的隐性缺失。
- [ ] 检查不可能的数值范围、单位混用和异常类别；不要直接把所有极端值删掉。
- [ ] 字符串时间转成 datetime，确认时区，再排序。
- [ ] 检查重复 timestamp、相邻时间差、缺少的时间点和采样频率。
- [ ] 区分“有这一行，但值缺失”和“整个时间点缺失”。
- [ ] 有夏令时的地区，检查 UTC 与本地时间的区别；本地一天可能有 23 或 25 小时。
- [ ] 标签有缺失时，不要把填出来的标签当真实观察值用于评分。

可以说：“我会先确认异常值是否符合业务定义，再决定处理方式。例如零发电量和负 residual load 可能都有实际意义。”

**4. 尽早设计数据切分，保留最终测试集**

- [ ] 时间序列按时间顺序划分，使用 expanding-window 或 rolling-window validation。
- [ ] 非时间序列若有同一用户、设备或地点的重复记录，考虑 group split。
- [ ] 只有记录近似独立且符合部署场景时，才用随机切分；分类任务按需要分层。
- [ ] 有标签窗口重叠或发布延迟时，按信息可用性设置 gap / purge，而不是机械套固定间隔。
- [ ] 用训练集做影响模型选择的 EDA，用验证集比较方案；测试集留到模型方案锁定后再用。
- [ ] 插补、标准化、类别编码的学习部分、特征筛选、PCA、季节性拟合都必须在每个训练折内部完成。

提醒：全表的基本质量检查与用全表学习模型参数是两回事。看到测试集成绩后再调模型，这个测试集就参与了选择，不能继续称为独立的最终验证。

**5. 做有目的的 EDA：每张图回答一个问题**

- [ ] Target 分布：是否偏态、有极端值、大量零值或类别不平衡？
- [ ] Predictor 分布：是否近乎常数、高基数、稀有类别、强偏态？
- [ ] Predictor 与 target 的关系：散点图、分组统计；是否有非线性、交互或不同群体的差异？
- [ ] 缺失是否集中在特定时间、群体或高低值附近？
- [ ] 时间序列画原始走势、滚动统计、小时/星期/月的平均轮廓，寻找趋势、季节性和结构变化。
- [ ] 用 ACF、频谱或日历回归补充季节性诊断；重复峰值本身不是因果关系或正式显著性证明。
- [ ] 用训练期内的较晚时间段检查规律是否稳定，避免把某一年的偶然事件当作可重复的年度规律。

应产出：三到五个明确发现，每个发现对应后续处理、特征或验证问题。

**6. 处理缺失和异常：按数据机制决定**

- [ ] 先报告缺失比例、连续缺口长度和发生位置，再选方法。
- [ ] 普通数值列可从训练集 median 开始；类别列可考虑单独的 missing 类别。
- [ ] 时间序列可比较有限长度的 forward fill、历史同小时/同星期匹配和其他仅依赖历史的方案。
- [ ] 长缺口不要无限 forward fill；考虑缺失标记、回退预测或明确剔除不可评分样本。
- [ ] 只使用预测时可得的信息；双向插值可能引入未来信息。
- [ ] 对确认的录入错误做修正；对真实极端事件考虑稳健方法、变换或单独分析。
- [ ] 如果数据没有缺失，明确报告“无需实际插补”，再说明未来发生缺失时的处理策略。

可以说：“我不会先机械地填均值；我会先看缺失发生的机制，以及填补时能用到哪些历史信息。”

**7. 诊断季节性与共线性，再决定是否处理**

- [ ] 季节性可以通过 calendar features 直接建模，也可以先调整季节项，或使用季节差分；不要求所有模型都先去季节性。
- [ ] 月份 dummy 与年度 Fourier 都可作为候选。选择取决于曲线形状、复杂度和验证表现。
- [ ] 如果减掉完整的“截距 + 趋势 + 季节项”拟合值，剩下的是调整后的残差，不是只去掉季节性的原始变量。
- [ ] 如果变换 target，写清楚如何逆变换到原始尺度。季节差分 z_t = y_t - y_(t-m) 需要把基准加回；多步预测时先检查该基准是否已知。
- [ ] 检查 predictor 之间的相关性、确定性公式、矩阵秩；VIF 可辅助诊断线性模型，但不要把 target 放进 predictor 的 VIF 矩阵。
- [ ] 线性模型可考虑删除冗余变量、Ridge / Elastic Net；需要时再考虑 PCA。
- [ ] 树模型不要求消除所有共线性，但相关特征会影响重要性解释。不要把相关性或特征重要性说成因果效应。

可以说：“季节性首先是预测信号，不一定是要删除的噪声；我会用验证结果判断应该显式建模还是先做调整。”

**8. 构造特征，检查每个特征的时间来源**

- [ ] 通用候选：有业务意义的比率、交互、类别编码，以及适用的数值变换。
- [ ] 时间序列候选：target 和其他变量的 lag、历史 rolling mean/std、小时、星期、月份、节假日等。
- [ ] 滚动统计必须结束在预测发出前的可用历史范围内；检查 shift 和窗口边界。
- [ ] 对每个 horizon，验证 lag 特征确实来自预测起点以前。
- [ ] 区分未来可预先知道的日历信息，与未来无法直接知道的天气、负荷、销量实际值；需要后者时只能使用当时可得的预测版本。
- [ ] 标签和特征行严格对齐，明确一行训练样本对应哪个预测起点、哪个目标时刻。
- [ ] 只预测 target，不把 target 本身、它的直接推导项或预测后才产生的字段意外放进特征。

应产出：最终 X 和 y 的含义、维度、时间范围，以及特征可用性检查。

**9. 先建 baseline，再比较少量模型**

- [ ] 回归：训练均值/中位数 baseline，再考虑 Ridge / Elastic Net 和树提升模型。
- [ ] 分类：简单类别 baseline、Logistic Regression，再考虑树模型；指标按类别不平衡和决策成本选。
- [ ] 时间序列：上一时点、上一天同小时、上一周同小时等合理 naive baseline，再考虑回归、boosting 或 ARIMA / SARIMA。
- [ ] 模型复杂度应与数据量、训练时间、可解释性需求相匹配，不必把所有模型跑一遍。
- [ ] 所有模型使用相同切分、预测任务、评分样本和主要指标。
- [ ] 用验证集或交叉验证做特征/超参数选择；同时比较不同折的稳定性、耗时和复杂度。

可以说：“我先检查模型能否稳定超过一个合理的简单基准，再判断增加复杂度是否值得。”

**10. 锁定方案，评估预测效果和失败场景**

- [ ] 锁定模型后，用训练集和验证集重新拟合，再在保留的测试集上评估。
- [ ] 回归报告 MAE、RMSE 等；说明量纲。目标有零值或接近零时谨慎使用 MAPE。
- [ ] 分类按任务报告 precision/recall、PR-AUC、ROC-AUC、校准或成本；阈值在验证集确定。
- [ ] 报告相对 baseline 的改善，而不是只给一个孤立分数。
- [ ] 画实际值与预测值、误差分布，并明确误差符号，如 prediction - actual。
- [ ] 按时间、horizon、群体、高峰/低谷等切片检查；查看系统性偏差和时间序列误差 ACF。
- [ ] 检查最差案例；如有预测区间，报告覆盖率和宽度，并说明其适用范围。
- [ ] 有时间再做简单 ablation，判断某组特征是否真正改善验证表现。

提醒：残差仍有自相关表示可能还有可利用的信息，但不等于模型没有预测价值。测试集上表现好也不证明未来没有结构变化。

**11. 交付可以复现、可以使用的预测流程**

- [ ] 保存 mapping、列顺序、预处理器、模型、随机种子、环境依赖和训练时间范围。
- [ ] 固定预测函数输入/输出格式，验证加载模型后的结果与保存前一致。
- [ ] 输出样例预测，并确认单位、时间戳、时区和逆变换正确。
- [ ] 如实说明假设、数据限制、发布延迟以及没有做过的比较。
- [ ] 提出下一步：新特征、更多回测、漂移监控、定期更新或缺失时的回退策略。

**最后两分钟：十个必查问题**

1. 我究竟预测什么、什么时候预测？
2. 每个 feature 在预测时真的拿得到吗？
3. 切分是否符合时间和群体结构？
4. 有没有用测试集参与插补、筛选、调参或选模型？
5. target 有没有误入 predictor，或者被其他字段直接泄漏？
6. 缺失、零值、负值、异常值和单位解释清楚了吗？
7. 所有模型是否与同一个合理 baseline 公平比较？
8. 最终结果是否回到了原始 target 的单位和尺度？
9. 除了整体分数，是否知道模型在哪里预测得不好？
10. 能否重新运行并得到相同结果，且清楚说出限制？

**面试开场，可以直接说**

“我会先明确预测目标、预测时点和可用信息，再检查数据结构、质量和时间索引。我会根据部署场景设计训练、验证和测试切分，并在训练数据上做 EDA，识别缺失、异常、季节性和冗余关系。之后建立简单 baseline，构造不泄漏未来信息的特征，并用一致的验证方案比较少量模型。最后锁定方案，在保留的测试集上评估整体表现和失败场景，交付可复现的 pipeline，并说明假设和改进方向。”

**English version: follow this sequence during the interview**

**1. Define the prediction task and information available at prediction time**

- [ ] Identify the unit of observation: a customer, transaction, device, or hour.
- [ ] Define the target, task type, units, and business decision.
- [ ] Specify when predictions are issued and how far ahead they must look.
- [ ] Check which features are available then, including publication delays.
- [ ] Agree on the primary metric and a meaningful baseline. If choosing the target yourself, justify its relevance.

What to say: “I will first define the target, prediction time, and evaluation criteria so that I can determine which features are available for a valid prediction.”

Time-series reminder: forecasting all 24 hours of tomorrow at one origin differs from updating a forecast every hour. The former cannot use tomorrow morning's observed values to predict tomorrow afternoon.

**2. Inspect the schema and understand the variables**

- [ ] Inspect shape, sample rows, data types, descriptive statistics, category frequencies, and cardinality.
- [ ] Read the data dictionary; verify units, sign conventions, and ID-to-name mappings.
- [ ] Identify keys and investigate duplicate rows before removing them.
- [ ] Distinguish IDs, numerical features, categories, timestamps, labels, and derived variables.
- [ ] For joins, check key uniqueness, join cardinality, and row counts before and after merging.

Deliverable: a variable dictionary with a feature-availability assessment.

**3. Check data quality and the time index**

- [ ] Check missing values, infinities, parsing failures, and hidden missing-value codes.
- [ ] Investigate impossible ranges, inconsistent units, and unexpected categories.
- [ ] Parse timestamps, establish the time zone, and sort the data.
- [ ] Check duplicate timestamps, consecutive time differences, missing time points, and sampling frequency.
- [ ] Distinguish missing values within existing rows from missing rows in the time index.
- [ ] Account for daylight saving time: a local calendar day can contain 23 or 25 hours.
- [ ] Do not evaluate predictions against artificially imputed labels.

What to say: “I will check whether unusual values are valid under the variable's business definition before treating them as errors.”

**4. Design the evaluation split early and reserve a final test set**

- [ ] For time series, use chronological splits and expanding- or rolling-window validation.
- [ ] Use group-aware splitting when the same customer, device, or location occurs repeatedly and the deployment task requires generalization to new groups.
- [ ] Use random splitting only when the independence assumptions and deployment setting support it; stratify classification splits when appropriate.
- [ ] Consider gaps or purging for overlapping label windows and information delays.
- [ ] Use training data for EDA that informs modeling, validation data for selection, and the test set for the locked final evaluation.
- [ ] Fit imputation, scaling, learned encodings, feature selection, PCA, and seasonal adjustments separately within each training fold.

Reminder: basic integrity checks across a dataset are different from learning model choices from held-out data. Once test performance guides tuning, that test set is no longer an independent final assessment.

**5. Perform EDA with specific questions**

- [ ] Inspect the target for skewness, outliers, excess zeros, or class imbalance.
- [ ] Inspect predictors for near-constant values, rare categories, high cardinality, and unusual distributions.
- [ ] Explore predictor–target relationships, nonlinearity, interactions, and subgroup differences.
- [ ] Examine whether missingness clusters by time, group, or value range.
- [ ] For time series, plot trajectories, rolling summaries, and hour-of-day, weekday, and month profiles.
- [ ] Use ACF, spectra, or calendar regression to support seasonal diagnosis; visual patterns alone do not establish causality or formal significance.
- [ ] Check stability in later periods within the training data, rather than assuming one year's pattern will repeat.

Deliverable: three to five findings, each linked to a preprocessing decision, feature, or validation question.

**6. Handle missing values and outliers according to their mechanism**

- [ ] Report missingness rates, gap lengths, and locations before choosing an approach.
- [ ] Consider training-set median imputation for numerical features and a missing category for categorical features.
- [ ] For time series, consider limited forward filling or historical seasonal matches, using only available observations.
- [ ] For long gaps, consider missingness indicators, fallback forecasts, or explicitly excluding unscorable samples.
- [ ] Avoid interpolation that uses information unavailable at the forecast origin.
- [ ] Correct confirmed errors; retain and analyze genuine extreme events or use suitable robust methods.
- [ ] If there are no missing values, report that no imputation was needed and describe a contingency policy.

What to say: “I will choose a missing-data strategy based on the missingness pattern and the information available at prediction time.”

**7. Diagnose seasonality and multicollinearity before deciding how to handle them**

- [ ] Seasonality can be modeled through calendar features, seasonal adjustment, or seasonal differencing; preprocessing it away is not mandatory.
- [ ] Month dummies and annual Fourier terms are both candidates; choose based on shape, complexity, and validation performance.
- [ ] Subtracting a fitted intercept, trend, and seasonal component produces an adjusted residual, not a series with only seasonality removed.
- [ ] If transforming the target, specify the inverse transformation. For seasonal differences, add the baseline back and verify that it is known at each forecast horizon.
- [ ] Check predictor correlations, deterministic identities, and matrix rank. VIF can support linear-model diagnosis; exclude the response from the predictor VIF matrix.
- [ ] Consider removing redundant features or using Ridge / Elastic Net; use PCA only when its tradeoffs are justified.
- [ ] Trees do not require eliminating every correlated feature, but correlation complicates feature-importance interpretation. Predictive importance is not causality.

What to say: “Seasonality is potentially useful predictive information. I will validate whether to model it directly or adjust for it first.”

**8. Engineer features and audit their availability**

- [ ] Consider meaningful ratios, interactions, categorical encodings, and appropriate numerical transformations.
- [ ] For time series, consider target and covariate lags, historical rolling means and standard deviations, and calendar features.
- [ ] Ensure rolling windows end within the history available at the prediction origin; check shifts and boundary conditions.
- [ ] Verify lag availability separately for each forecast horizon.
- [ ] Distinguish known future calendar information from unknown future realized covariates. Use only the forecast versions of external variables available at the origin.
- [ ] Verify feature–label alignment and record the forecast origin and target time for each sample.
- [ ] Exclude accidental target leakage, including direct target derivatives and post-outcome fields.

Deliverable: an explicit definition of X and y, their dimensions and time ranges, and checks of feature availability.

**9. Establish baselines and compare a small set of models**

- [ ] Regression: a training mean/median baseline, then regularized linear models and possibly gradient boosting.
- [ ] Classification: a simple class baseline, Logistic Regression, and possibly tree models; choose metrics around imbalance and decision costs.
- [ ] Time series: appropriate persistence or seasonal-naive forecasts, then regression, boosting, or ARIMA / SARIMA where justified.
- [ ] Match complexity to sample size, runtime, and interpretability requirements.
- [ ] Compare models on identical splits, forecast tasks, scoring samples, and primary metrics.
- [ ] Select features and hyperparameters using validation only; consider stability across folds, runtime, and complexity as well as average accuracy.

What to say: “I will first establish whether the model consistently improves on a meaningful simple baseline before adding complexity.”

**10. Lock the approach and assess accuracy and failure modes**

- [ ] After selection, refit on training and validation data and evaluate on the reserved test set.
- [ ] For regression, report metrics such as MAE and RMSE with units; be cautious with MAPE when targets are zero or close to zero.
- [ ] For classification, select relevant precision/recall, PR-AUC, ROC-AUC, calibration, or cost measures; choose decision thresholds on validation data.
- [ ] Report improvement over the baseline, not just a standalone score.
- [ ] Plot actual versus predicted values and error distributions; state the error sign convention.
- [ ] Inspect performance by time, horizon, subgroup, and extreme conditions; check bias and time-series error autocorrelation.
- [ ] Investigate the worst cases. If using prediction intervals, report coverage and width with appropriate limitations.
- [ ] If time permits, use simple ablations to test whether feature groups improve validation performance.

Reminder: autocorrelated prediction errors can indicate remaining information, but do not automatically invalidate a useful forecast. Good historical performance does not rule out future structural changes.

**11. Deliver a reproducible and usable pipeline**

- [ ] Save mappings, feature order, preprocessing objects, model, random seeds, dependencies, and training period.
- [ ] Specify prediction input/output formats and verify save/reload consistency.
- [ ] Provide sample predictions with correct units, timestamps, time zones, and inverse transformations.
- [ ] State assumptions, limitations, availability delays, and comparisons that have not been performed.
- [ ] Propose next steps: additional features, further backtesting, drift monitoring, model updates, or fallback handling.

**Final two-minute check**

1. What am I predicting, and when is the prediction issued?
2. Is every feature actually available at that time?
3. Does the split respect temporal and group structure?
4. Has test information influenced preprocessing, feature selection, tuning, or model choice?
5. Is the target leaked directly or through a derived field?
6. Have I explained missing values, zeros, negatives, extremes, and units?
7. Are all models compared fairly against a meaningful baseline?
8. Are final predictions on the original target scale?
9. Do I know where the model performs poorly, beyond its aggregate score?
10. Can I reproduce the result and explain its limitations?

**An opening statement you can use**

“I would first clarify the prediction target, the prediction time, and the information available then. Next, I would inspect the schema, data quality, and time index, and design a split that reflects deployment. I would use training-data EDA to investigate missingness, outliers, seasonality, and redundant relationships. Then I would build a simple baseline, engineer features without leaking future information, and compare a small set of models under the same validation scheme. Finally, I would lock the approach, assess accuracy and failure modes on the reserved test set, and deliver a reproducible pipeline with clear assumptions and limitations.”

**如果面试时间有限 / If time is limited**

优先顺序：明确任务和切分 → 数据质量检查 → 简单 baseline → 一个合理候选模型 → 误差检查和结论。复杂季节分解、PCA、大规模调参和额外模型，只有在诊断支持且时间允许时再做。不要为了展示更多方法而没有完成可靠的预测闭环。

Priority: define the task and split → check data quality → establish a simple baseline → fit one reasonable candidate → inspect errors and explain the result. Add complex decomposition, PCA, extensive tuning, or additional models only when justified and time permits. A complete, trustworthy prediction workflow is more valuable than a long list of unfinished techniques.
