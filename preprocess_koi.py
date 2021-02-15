# -*- coding: utf-8 -*-
import pandas as pd

from pathlib import Path
from utils import load_dataset, fill_na_mean
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



'''
implements the functionality to preprocess the KOI data
'''
class PreprocessKOI():
    def __init__(self):
        self.habitable_planets = None
        self.non_habitable_planets = None
        self.all_data = None
        self.target_columns = None
        
        
    '''
    imports the habitable planet data 
    '''
    def get_habitable_planets(self, csv_path):
        self.habitable_planets = pd.read_csv(csv_path, sep=',')
        
        
    '''
    imports the non-habitable planet data 
    '''
    def get_non_habitable_planets(self, csv_path):
        self.non_habitable_planets = pd.read_csv(csv_path, sep=',')
    
    
    '''
    gets the all of the columns in the dataset
    '''
    def get_column_names(self, csv_path):
        with open(csv_path, 'r') as data:
            self.target_columns = [value.rstrip("\n") for value in data.readline().split(',')]
    
    
    '''
    appends y column to datasets and gives them a class
    1 = planet is potentially habitable
    0 = planet is not habitable
    '''
    def append_labels(self, val):
        if val == 1:
            self.habitable_planets['y'] = 1
        else:
            self.non_habitable_planets['y'] = 0
            
        
    '''
    combines all of the rows and shuffles them
    '''
    def combine_data(self):
        temp_hab = self.habitable_planets
        temp_non_hab = self.non_habitable_planets
        temp_combined = pd.concat([temp_hab, temp_non_hab])
        temp_combined = temp_combined.sample(frac=1)
        
        self.all_data = temp_combined
        

    '''
    specify the list of features to keep
    '''
    def keep_features(self, keep_list):
        drop_list = [value for value in self.target_columns if value not in [c.rstrip("\n") for c in keep_list]]
        self.target_columns = [value for value in self.target_columns if value in [c.rstrip("\n") for c in keep_list]]
        self.all_data.drop(drop_list, axis=1, inplace=True)
    
    
    '''
    specify the list of features to drop
    '''
    def drop_features(self, drop_list):
        self.target_columns = [value for value in self.target_columns if value not in [c.rstrip("\n") for c in drop_list]]
        self.all_data.drop(drop_list, axis=1, inplace=True)

        
    '''
    use SelectFromModel feature selection algorithm to select the best features
    '''
    def smart_select(self, thresh):
        cols = self.target_columns[:] #copy the columns
        cols.append('y') #appened the y column
        sliced_hab = self.habitable_planets[cols].head(50) #take the first 10 rows of hab planets
        sliced_non_hab = self.non_habitable_planets[cols].sample(frac=1).head(1000) #take 200 rows of the shuffled non-hab planets
        sliced_combined = pd.concat([sliced_hab, sliced_non_hab]) #combine them
        
        x, y = load_dataset(sliced_combined)
        x, _ = fill_na_mean(self.target_columns, x, x)
        
        feat_selector = SelectFromModel(estimator=LogisticRegression(), threshold=thresh).fit(StandardScaler().fit_transform(x), y)
        
        #get the list of features
        #then get the ones that are 'false', meaning they are determined to be poor
        #drop those columns
        list_of_feats = list(feat_selector.get_support())[:]
        drop_list = []
        for f in range(len(list_of_feats)):
            if not list_of_feats[f]:
                drop_list.append(self.target_columns[f])
        
        self.drop_features(drop_list)

    
#    '''
#    remove some fraction of the negative class, since there are way more 
#    instances of them than of the positive class
#    frac_to_remove is between 0 and 1 inclusively
#    '''
#    def undersample(self, frac_to_remove):
#        total = len(self.non_habitable_planets)
#        remove = int(float(total) * frac_to_remove)
#        print(total)
#        drop_indices = np.random.choice(self.non_habitable_planets.index, remove, replace=False)
#        subset = self.non_habitable_planets.drop(drop_indices)
#        self.non_habitable_planets = subset
#        print(len(self.non_habitable_planets))
        
        
    '''
    save the processed data to /data/processed_data.csv
    '''
    def write_csv(self, exp_name=''):
        self.all_data.to_csv(exp_name + 'processed_data.csv')
    
    
    '''
    performs the base preprocessing functions:
        gets all of the habitable planets
        gets all of the non-habitable planets
        gets all the headers as a list
        appends y column and 1 to the hab planet dataset
        appends y column and 0 to the non-hab planet dataset
        combines both datasets together and shuffles the result
    '''
    def base_preprocessing(self):
        hab_planets = Path(__file__).parent / "../data/habitable_planets_detailed_list.csv"
        non_hab_planets = Path(__file__).parent / "../data/non_habitable_planets_confirmed_detailed_list.csv"
        
        #this is a base set of features that aren't needed:
        init_remove = ['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 
                         'koi_vet_stat', 'koi_vet_date', 'koi_pdisposition', 'koi_disp_prov',
                         'koi_comment', 'koi_eccen', 'koi_eccen_err1', 'koi_eccen_err2', 
                         'koi_longp', 'koi_longp_err1', 'koi_longp_err2', 'koi_ingress', 
                         'koi_ingress_err1', 'koi_ingress_err2', 'koi_fittype', 'koi_sma_err1',
                         'koi_sma_err2', 'koi_incl_err1', 'koi_incl_err2', 'koi_limbdark_mod',
                         'koi_ldm_coeff4', 'koi_ldm_coeff3', 'koi_parm_prov', 'koi_trans_mod', 
                         'koi_model_dof', 'koi_model_chisq', 'koi_datalink_dvr', 'koi_datalink_dvs',
                         'koi_sage', 'koi_sage_err1', 'koi_sage_err2', 'koi_sparprov',
                         'koi_teq_err1', 'koi_teq_err2', 'koi_tce_delivname', 'koi_quarters']
        
        self.get_habitable_planets(hab_planets)
        self.get_non_habitable_planets(non_hab_planets)
        self.get_column_names(hab_planets)
        self.append_labels(1)
        self.append_labels(0)
        self.combine_data()
        self.drop_features(init_remove)
        
        
if __name__ == "__main__":  
    pr = PreprocessKOI()
    pr.base_preprocessing()
    test = pr.all_data.to_numpy()