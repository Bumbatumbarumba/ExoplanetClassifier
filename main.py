# -*- coding: utf-8 -*-
import utils

from naive_bayes import NaiveBayes
from log_reg import LogReg
from preprocess_koi import PreprocessKOI



'''
provide an experiment name, a list of features, and an operation
to perform with the feature list (either drop, keep, or select)

the results are saved in ../data/<exp_name>/
'''
def exp(exp_name, feat_list=[], feat_ops=''):
    print('=================================start of ' + exp_name)
    pr = PreprocessKOI()
    pr.base_preprocessing()
    
    if feat_ops == 'drop':
        pr.drop_features(feat_list)
    elif feat_ops == 'keep':
        pr.keep_features(feat_list)
    elif feat_ops == 'select':
        pr.smart_select(0.8)
    
    nb = NaiveBayes()
    nb.fit(pr)
    nb.predict()
    
    lr = LogReg()
    lr.fit(pr)
    lr.predict()
    
    utils.save_data_and_evals(pr, nb, lr, exp_name)
    print("=================================completed " + exp_name + "\n\n\n")
    
    
    
'''
the main set of experiments to conduct

modify this and then modify if __name__ == "__main__":
to conduct a desired set of experiments
'''
def main():
    #exp1 is no changes to the dataset
    exp("exp1")
    
    #exp2 is my educated guess at a good feature set
    exp('exp2', 
        feat_list = ['koi_period', 'koi_srho', 'koi_prad', 'koi_sma', 'koi_incl', 'koi_duration',
                             'koi_teq', 'koi_dor', 'koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass',
                             'koi_ror', 'koi_depth', 'koi_count', 'koi_num_transits', 'koi_smet'], 
        feat_ops='keep')
    
    #exp3 will be the result after using a feature selection alg
    exp('exp3', feat_ops='select')
    
    #exp4 is a narrowed down set from exp2
    exp('exp4', 
        feat_list = ['koi_period', 'koi_prad', 'koi_sma', 'koi_incl',
                     'koi_teq', 'koi_dor', 'koi_steff', 'koi_srad', 'koi_smass',
                     'koi_count'], 
        feat_ops='keep')

    

'''
running this will result in 10 runs of exp1 to 4 to create a set of results 
to analyze and determine the best resulting model.
'''
def find_best():    
    count = 0
    while count < 10:
        #exp1 is no changes to the dataset
        exp('bot/exp1_' + str(count))
        
        #exp2 is my educated guess at a good feature set
        exp('bot/exp2_' + str(count), 
            feat_list = ['koi_period', 'koi_srho', 'koi_prad', 'koi_sma', 'koi_incl', 'koi_duration',
                                 'koi_teq', 'koi_dor', 'koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass',
                                 'koi_ror', 'koi_depth', 'koi_count', 'koi_num_transits', 'koi_smet'], 
            feat_ops='keep')
        
        #exp3 will be the result after using a feature selection alg
        exp('bot/exp3_' + str(count), feat_ops='select')
        
        #exp4 is a narrowed down set from exp2
        exp('bot/exp4_' + str(count), 
            feat_list = ['koi_period', 'koi_prad', 'koi_sma', 'koi_incl',
                         'koi_teq', 'koi_dor', 'koi_steff', 'koi_srad', 'koi_smass',
                         'koi_count'], 
            feat_ops='keep')
        count += 1
        
    
'''
if you comment out main() and uncomment find_best(), it will perform
10 runs of each experiment and save them to ../data/bot/
'''
if __name__ == "__main__":
    main()
    #find_best()