import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LassoCV, Lasso
from sklearn.model_selection import train_test_split

class OpioidAnalysis:
    def __init__(self, param: str) -> None:
        self.state_lst = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID',
        'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC',
        'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD',
        'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
        self.results = []
        self.param = param
        self.model = None

    def get_full_df(self, path: str) -> pd.DataFrame:
        """
        function to import data from .csv file and return pandas df

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
        
        assert state in self.state_lst, "Value Error: state abbreviation provided not valid"
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

    def write_results(self, state: str, year: int):
        """
        function to write to results property

        inputs
        -----
        state: -> state abbreviation i.e.'WV'
        year: -> year of analysis i.e. 2009 
        """
        features = zip(self.model.feature_names_in_, list(self.model.coef_))
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
        output_df = pd.DataFrame(self.results)
        path = path + f'\{self.param}_results.csv'
        output_df.to_csv(path)    

def run_analysis(param: str):
    # instantiate analysis class
    op_analysis = OpioidAnalysis(param=param)
    
    # define path to data and import
    input_file_path = '.\Datasets\\Analytic File 3-31-21 DIB.csv'
    full_df = op_analysis.get_full_df(path=input_file_path)
    
    # get list of states
    states = full_df['F12424'].unique()
    year = full_df['YR'].unique()

    # for each state
    for st in states:
        if st != 'DC': # DC only has one county
            # filter df to state
            state_df = op_analysis.get_state_df(df=full_df, state=st)
            # for each year
            for yr in year:
                print(f'evaluating state {st} in year {yr} for independent param {op_analysis.param}')
                # filter to single year
                state_yr_df = state_df[state_df['YR'] == yr]
                # define columns to drop
                drop_cols = ['YR', 'F00002', 'F12424', 'F00010' ,'F04437', 'PILL_QUART', 'ORD_DEATHS_NOIMP', 'ORD_CDR_NOIMP', 'CDR_NOIMP', 'CANCER_DEATHS_NOIMP', 'CANCER_CDR_NOIMP']
                # drop unnecessary columns
                filtered_state_df = op_analysis.filter_df(df=state_yr_df, cols=drop_cols)
                # standardize data
                standardised_df = op_analysis.standardise_df(df=filtered_state_df)
                # run lasso cv
                if len(standardised_df) < 5:
                    op_analysis.run_lasso(df = standardised_df, model=LassoCV(max_iter = 5000, eps=5e-2, cv = 3))
                else:
                    op_analysis.run_lasso(df = standardised_df, model=LassoCV(max_iter = 5000, eps=5e-2, cv = 5))         
                # add features to output
                op_analysis.write_results(state=st, year=yr)
        else: # for DC
             for yr in year:
                print(f'evaluating state {st} in year {yr} for independent param {op_analysis.param}')
                # filter to single year
                state_yr_df = state_df[state_df['YR'] == yr]
                # define columns to drop
                drop_cols = ['YR', 'F00002', 'F12424', 'F00010' ,'F04437', 'PILL_QUART', 'ORD_DEATHS_NOIMP', 'ORD_CDR_NOIMP', 'CDR_NOIMP', 'CANCER_DEATHS_NOIMP', 'CANCER_CDR_NOIMP']
                # drop unnecessary columns
                filtered_state_df = op_analysis.filter_df(df=state_yr_df, cols=drop_cols)
                # standardize data
                standardised_df = op_analysis.standardise_df(df=filtered_state_df)
                # run lasso cv
                op_analysis.run_lasso(df = standardised_df, model=Lasso(alpha=0.1))       
                # add features to output
                op_analysis.write_results(state=st, year=yr)                                               
    # write to csv     
    op_analysis.output_results(path='.\Analysis\\results')

if __name__ == "__main__":
    run_analysis(param='PCPV')
    run_analysis(param='ORD_DEATHS')