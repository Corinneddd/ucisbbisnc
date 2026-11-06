# TenneT 分析与 24 小时预测

先打开 analysis_report_zh.html 阅读完整中文结果，或打开 tennet_analysis.ipynb 逐步检查。

环境：Python 3.12；`python -m pip install -r requirements.txt`。

完整重跑：
`python tennet_pipeline.py --data /path/to/hourly.csv --mapping /path/to/mapping.csv --out results`

预测：
`python predict_24h.py --model results/production_model.joblib --data /path/to/hourly.csv --mapping /path/to/mapping.csv --origin 2022-12-31T00:00:00Z --out forecast_24h.csv`

保持 tennet_pipeline.py 与 predict_24h.py 在同一目录。原始两个 CSV 由用户提供，压缩包包含它们的重命名数据和完整字段字典，原文件未改动。若在其他电脑重跑，修改输入路径。

窗口：每日 UTC 00:00 一次预测接下来 24 个小时；柏林标签用于日历特征。原始数值单位 MWh。模型假设上一小时数据在起点已发布。

evaluation_model.joblib 使用 2022 年以前的训练标签；production_model.joblib 在最终测试后用全部可用完整日重训。不可用生产模型重算并宣称独立 2022 测试成绩。endpoint_24h_forecast_demo.csv 是历史末端演示，不是当前日期的未来预测。

最终模型、指标、所有图表与方法限制均见中文报告。Notebook 含真实执行输出；代码使用固定随机种子和记录的依赖版本。
