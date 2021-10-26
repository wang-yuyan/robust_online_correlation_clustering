# Code for "Robust Online Correlation Clustering"

<img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" width=200>

Published at the following Github repository: 
https://github.com/wang-yuyan/robust_online_correlation_clustering

The ML Code Completeness Checklist consists of five items:

1. **Specification of dependencies**
2. **Training code** 
3. **Evaluation code**


We explain each item on the checklist in detail blow. 

#### 1. Specification of dependencies
The remaining of the codes depend on data_preprocess.py as that produces the training datasets using the raw data downloaded. Check the main paper for the link and datasets. pivot.py implements pivot algorithm, with uniform advice or corrupted uniform advice. temporal_advice.py is the ML oracle used to generate advice from data and depends on pivot.py. Then all three files ending with "_analysis" (historical_data_output_analysis, output_analysis and robustness_analysis) are meant for running experiments, collecting data and output table/figures.

#### 2. Training code
pivot and pivot_temporal.py are the training codes.

#### 3. Evaluation code
The output_analysis and historical_data_output_analysis files are meant for collecting and analyzing the random advice and temporal advice semi-online model, respectively. They would generate data tables in designated repositories.




