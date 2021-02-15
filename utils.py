import os
import shutil
import pandas as pd
#import matplotlib.pyplot as plt
import sklearn.metrics as sm


    
'''
we do this action in the util because we want to 
fill na with mean AFTER doing the split. this
prevents data from leaking over from each set and
so give a better result
'''
def fill_na_mean(cols, x_train, x_test):
    #fill empty vals in x train set with mean
    df_x_train = pd.DataFrame(x_train,
                              index=range(x_train.shape[0]),
                              columns=cols)
    df_x_train.fillna(df_x_train.mean(), inplace=True)
    # #fill empty vals in x test set with mean
    df_x_test = pd.DataFrame(x_test,
                              index=range(x_test.shape[0]),
                              columns=cols)
    df_x_test.fillna(df_x_test.mean(), inplace=True)

    return df_x_train.to_numpy(), df_x_test.to_numpy()


'''
converts the pandas dataframe to a numpy array
storing the label data and the feature data
'''
def load_dataset(data):
    np_data_label = data['y'].to_numpy()
    no_y = data.drop('y', axis=1)
    np_data_features = no_y.to_numpy()
    
    return np_data_features, np_data_label
   
    
'''
saves the evaluation data of both models to../data/<exp>/,
as well as the true and predicted labels for the test set
'''
def save_data_and_evals(pr, nb, lr, exp):
    exp_dir = '../data/' + exp + '/'
    
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    
    os.mkdir(exp_dir)
    pr.write_csv(exp_name=exp_dir)
    print('NAIVE BAYES EVALUATION')
    nb.save_txt(exp_dir)
    save_evaluations(nb, exp_dir, 'nb_evaluation', 'nb')
    #plot_con_mat(nb, 'nb')
    
    print('\nLOGISTIC REGRESSION EVALUATION')
    lr.save_txt(exp_dir)
    save_evaluations(lr, exp_dir, 'lr_evaluation', 'lr')
    #plot_con_mat(lr, 'lr')    
    
 
'''
saves the evaluation results to ../data/<exp_name>/<save_name>.txt
and prints them to the console
'''    
def save_evaluations(clf, exp, save_name, exp_type):
    recall = calculate_recall(clf)
    precision = calculate_precision(clf)
    accuracy = calculate_accuracy(clf)
    #num_points, num_mislabeled, percent = get_num_mislabeled(clf)
    con_mat = confusion_matrix(clf)
    
    eval_loc = exp + save_name + '.txt'
    #con_mat_loc = exp + save_name + '.png'
    
    with open(eval_loc, 'a') as evals:
        evals.write('confusion matrix: \n' + str(con_mat) + '\n')
        evals.write('recalls is: ' + str(recall) + '\n')
        evals.write('precision is: ' + str(precision) + '\n')
        evals.write('accuracy is: ' + str(accuracy) + '\n')
        #evals.write('number mislabeled out of %d points : %d (%d percent)\n' % (num_points, num_mislabeled, percent))
    
    
    #plt.savefig(con_mat_loc)
    
    print('confusion matrix: \n' + str(con_mat))
    print('recalls is: ' + str(recall))
    print('precision is: ' + str(precision))
    print('accuracy is: ' + str(accuracy))
    #print('number of mislabeled points out of a total %d points : %d (%d percent)' % (num_points, num_mislabeled, percent))
    print('saved evaluation results to ' + eval_loc)


'''
generates a textual confusion matrix
'''
def confusion_matrix(nb): 
    true_labels = nb.y_test
    pred_labels = nb.y_hat
    con_mat = sm.confusion_matrix(true_labels, pred_labels)
    
    formatted_con_mat = 'pred neg, pred pos\n'
    formatted_con_mat += "    " + str(con_mat[0][0]) + '\t    ' + str(con_mat[0][1]) + '   true neg\n'
    formatted_con_mat += "    " + str(con_mat[1][0]) + '\t    ' + str(con_mat[1][1]) + '   true pos'
    
    return formatted_con_mat
    

'''
calculates the precision
'''
def calculate_precision(nb):
    true_labels = nb.y_test
    pred_labels = nb.y_hat
    return sm.precision_score(true_labels, pred_labels, pos_label=1)


'''
calculates the accuracy
'''
def calculate_accuracy(nb):
    true_labels = nb.y_test
    pred_labels = nb.y_hat
    return sm.accuracy_score(true_labels, pred_labels)


'''
calculates the recall
'''
def calculate_recall(nb):
    true_labels = nb.y_test
    pred_labels = nb.y_hat
    return sm.recall_score(true_labels, pred_labels, pos_label=1)
    

#'''
#generates the confusion matrix as a plt
#'''
#def plot_con_mat(clf, exp_type):
#    con_mat = None
#    
#    if exp_type == 'nb':
#        con_mat = sm.plot_confusion_matrix(clf.gnb,
#                                       clf.x_test,
#                                       clf.y_test,
#                                       cmap=plt.cm.Blues)
#        con_mat.ax_.set_title('Naive Bayes')
#    elif exp_type == 'lr':
#        con_mat = sm.plot_confusion_matrix(clf.lr,
#                                       clf.x_test,
#                                       clf.y_test,
#                                       cmap=plt.cm.Blues)
#        con_mat.ax_.set_title('Logistic Regression')
#        
#    #plt.show()
#    
#    return con_mat
#
#
#'''
#gets the number of mislabeled instances
#'''
#def get_num_mislabeled(nb):
#    num_points = nb.x_test.shape[0]
#    num_mislabeled = (nb.y_test != nb.y_hat).sum()
#    percent = num_mislabeled/num_points*100
#    
#    return num_points, num_mislabeled, percent

