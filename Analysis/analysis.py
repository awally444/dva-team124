import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso

class OpioidAnalysis:
    def __init__(self, param: str, all_years: bool) -> None:
        self.state_map = {'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AZ': 'Arizona', 
                          'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DC': 'Washington D.C.', 
                          'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'IA': 'Iowa', 'ID': 'Idaho',
                          'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts', 
                          'MD': 'Maryland', 'ME': 'Maine', 'MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri', 'MS': 'Mississippi', 
                          'MT': 'Montana', 'NC': 'North Carolina', 'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 
                          'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 
                          'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 
                          'VA': 'Virginia', 'VT': 'Vermont', 'WA': 'Washington', 'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming'}
        self.results = []
        self.param = param
        self.model = None
        self.all_years = all_years
        self.score = None

    def get_full_df(self, path: str) -> pd.DataFrame:
        """
        function to import data from .csv file and return pandas dfcon

        inputs
        -----
        path: -> path to data file

        outputs
        -----
        df: -> pandas dataframe from csv
        """
        # import opiod prescription data
        df = pd.read_csv(path, encoding='unicode_escape')
        return df

    def get_state_df(self, df: pd.DataFrame, state: str) -> pd.DataFrame:
        """
        function to return subset of full dataset
        subset is for a single state with unnecessary columns removed

        inputs
        -----
        df: -> dataframe containing data for all states
        state: -> state abbreviation of interest e.g. 'WV' (West Virginia)

        outputs
        -----
        state_df: -> pandas dataframe filtered for a single state with unnecessary columns removed 
        """
        
        assert state in self.state_map.keys(), "Value Error: state abbreviation provided not valid"
        # filter df to single state
        state_df = df.loc[df['F12424'] == state]
        return state_df

    def filter_df(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        function to remove columns from dataset

        inputs
        -----
        df: -> original dataframe
        cols: -> columns to remove from dataset

        outputs
        -----
        filtered_df: -> df with columns removed
        """
        # create list of columns to keep
        cols_to_keep = [x for x in list(df.columns) if x not in cols]
        # remove cols from df
        filtered_df = df[cols_to_keep]
        return filtered_df.dropna()

    def standardise_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        function to standardize dataset using ScikitLearn Standard Scaler

        inputs
        -----
        df: -> raw dataframe

        outputs
        -----
        standardised_df: -> dataframe after standardising values
        """
        # get the list of column names out now
        col_names = list(df.columns)
        # instantiate scaler and standardise values
        scaler = StandardScaler()
        model = scaler.fit(df)
        scaled_results = model.transform(df)
        # conver to df so that we can keep the feature names together
        standardised_df = pd.DataFrame(scaled_results, columns=col_names)
        return standardised_df

    def run_lasso(self, df: pd.DataFrame, model):
        """
        function to run lassocv analysis on data
        
        inputs
        -----
        df: -> standardised dataframe
        model: -> model for analysis i.e. LassoCV(max_iter = 5000) 
        """
        # define dependent and independent variables
        independent_cols = [x for x in list(df.columns) if x != self.param]
        X = df[independent_cols]
        Y = df[self.param]
        # run lasso cv
        self.model = model
        self.model.fit(X, Y)
        self.score = self.model.score(X, Y)

    def write_results(self, state: str, year=None):
        """
        function to write to results property

        inputs
        -----
        state: -> state abbreviation i.e.'WV'
        year: -> year of analysis i.e. 2009 
        """
        features = zip(self.model.feature_names_in_, list(self.model.coef_))
        if self.all_years:
            features_df = {'State': state}
            for f,c in features:
                features_df[f] = c
            features_df["R2"] = self.score
            self.results.append(features_df)
        else:
            features_df = {'State': state, 'Year': year}
            for f,c in features:
                features_df[f] = c
            self.results.append(features_df)

    def output_results(self, path: str):
        """
        function to write results to .csv file

        inputs
        -----
        path: -> path to directory where file will be saved
        """
        # create pandas df and create state name column
        output_df = pd.DataFrame(self.results)
        output_df.rename(columns={'State': 'State Abbv'}, inplace=True)
        output_df['State'] = output_df.apply(
            lambda x: self.state_map[x['State Abbv']], axis=1
        )
        # write out full table, with each column a features
        if self.all_years:
            output_df.to_csv(path + f'\\{self.param}_results_all_years.csv')
            # melt df for easier analysis
            ids = ['State Abbv', 'State']
            vals = [x for x in output_df.columns if x not in ids]
            output_df_melted = pd.melt(output_df, id_vars=ids, value_vars=vals)
            output_df_melted.to_csv(path + f'\\{self.param}_melted_results_all_years.csv')    
        else:
            output_df.to_csv(path + f'\\{self.param}_results.csv')
            # melt df for easier analysis
            ids = ['State Abbv', 'State', 'Year']
            vals = [x for x in output_df.columns if x not in ids]
            output_df_melted = pd.melt(output_df, id_vars=ids, value_vars=vals)
            output_df_melted.to_csv(path + f'\\{self.param}_melted_results.csv')    

def analysis(param: str, all_years: bool):
    """
    function to run analyis

    input
    -----
    param: -> feature from dataset to predict
    all_years: -> if true, analysis will group all years together, else will perform regression for each year independently
    """
    def run_analysis(df: pd.DataFrame, state: str):
        """
        helper function to analysis method
        """
        # filter df to state
        state_df = op_analysis.get_state_df(df=df, state=state)
        # define columns to drop
        drop_cols = ['YR', 'F00002', 'F12424', 'F00010' ,'F04437', 'PILL_QUART', 'ORD_DEATHS_NOIMP', 'ORD_CDR_NOIMP', 'CDR_NOIMP', 'CANCER_DEATHS_NOIMP', 'CANCER_CDR_NOIMP']
        drop_cols.extend(['F04538', 'F04542', 'ORD_CDR'])
        if param == 'PCPV':
            drop_cols.append('ORD_DEATHS')  # drop ord deaths as factor
            
        # drop unnecessary columns
        filtered_state_df = op_analysis.filter_df(df=state_df, cols=drop_cols)
        # standardize data
        standardised_df = op_analysis.standardise_df(df=filtered_state_df)
        if st != 'DC': # DC only has one county
            # run lasso cv
            if len(standardised_df) < 5:
                op_analysis.run_lasso(df = standardised_df, model=LassoCV(max_iter = 5000, eps=5e-2, cv = 3))
            else:
                op_analysis.run_lasso(df = standardised_df, model=LassoCV(max_iter = 5000, eps=5e-2, cv = 5))
        else:
            op_analysis.run_lasso(df = standardised_df, model=Lasso(alpha=0.5))
        # add features to output
        if op_analysis.all_years:
            op_analysis.write_results(state=st)
        else:
            op_analysis.write_results(state=st, year=yr)
    # instantiate analysis class
    op_analysis = OpioidAnalysis(param=param, all_years=all_years)
    
    # define path to data and import
    input_file_path = '.\Datasets\\Analytic File 3-31-21 DIB.csv'
    # input_file_path = '../Datasets/Analytic File 3-31-21 DIB.csv'
    full_df = op_analysis.get_full_df(path=input_file_path)
    
    # get list of states
    states = full_df['F12424'].unique()
    year = full_df['YR'].unique()

    # for each state
    for st in states:
        if all_years:
            print(f'evaluating state {st} for all years for independent param {op_analysis.param}')
            run_analysis(df=full_df, state=st)
        else:
            # for each year
            for yr in year:
                print(f'evaluating state {st} in year {yr} for independent param {op_analysis.param}')
                # filter to single year
                yr_df = full_df[full_df['YR'] == yr]
                run_analysis(df=yr_df, state=st)                                        
    
    # write to csv     
    op_analysis.output_results(path='.\\Analysis\\results')
    # op_analysis.output_results(path='../Analysis/results')   # mac path

if __name__ == "__main__":
    for p in ['PCPV', 'ORD_DEATHS']:
        analysis(param=p, all_years = True)
        analysis(param=p, all_years = False)