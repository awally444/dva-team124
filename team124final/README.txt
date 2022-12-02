DESCRIPTION
The code found in this directory can be run locally to recreate the analysis completed as part of the project. The directory contains the following files:
 - analysis.py -> python file responsible for creating linear regression models for each state using LASSO regression. The file is currently configured to perform analysis on Per-Capita Opioid Pill Volume (PCPV) and Opioid Related Deaths (ORD_DEATHS).
 - featurelabels.csv -> csv file mapping feature label codes to descriptions (e.g. F11984 -> Population estimate)
 - code.txt -> documentation for code portion of assignment 

The interactive visual tool is hosted on Tableau Public and can be found at the following link: https://public.tableau.com/app/profile/christopher.nelson4254/viz/SpatialandTemporalAnalysisofTrendsinOpioidAbusewithintheUS/OpiodProject?publish=yes 

INSTALLATION
The file 'analysis.py' requires the following libraries installed:
 - pandas
 - sklearn
Both can be installed using pip in the command prompt, 'pip install pandas sklearn'

Before running the analysis, the dataset 'Analytic File 3-31-21 DIB.csv' should be downloaded from https://data.mendeley.com/datasets/dwfgxrh7tn/6 and placed in the 'CODE' directory

EXECUTION
Once the required libraries are installed and the data file placed in the 'CODE' directory, the 'analysis.py' file can be executed to perform the analysis, which can be done in the command prompt via 'python .\CODE\\analysis.py'

As the execution is ongoing, messages will be printed displaying the state currently under evaluation and for what dependent feature, e.g. 'evaluating state AL for all years for parameter PCPV'. 

Once the analysis is complete, the results will be available within .csv files 'PCPV_results_all_years.csv' and 'ORD_DEATHS_results_all_years.csv'. The analysis is completed based on combining all years worth of data into one, denoted by the '...all_years' in the file name. Other features can be used as the dependent variable by changing the list on line 227 in analysis.py