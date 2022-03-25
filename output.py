from datetime import datetime
from matplotlib.pyplot import spy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import matplotlib as plt
import scipy.sparse as sparse

def show_sparse_matrix(csr_matrix):

    #print('Sparse matrix: ', csr_matrix)
    #print('Sparse matrix in array: ', csr_matrix.toarray())
    print('nnz of sparse matrix', csr_matrix.nnz)
    print("indptr:", csr_matrix.indptr)
    print("indices:", csr_matrix.indices)
    print("data:", csr_matrix.data)



    #spy(csr_matrix, markersize=1)

    #plt.show()

def print_table_header():
    print ("Algorithm   \tPrecision\t\tRecall/TPR\t\tFPR\t\tAccuracy\t\tF-Score")

def print_results_from_matrix(name, matrix):
    TP, FP = matrix[0]
    FN, TN = matrix[1]
    Precision = (TP * 1.0) / (TP + FP)
    Recall = (TP * 1.0) / (TP + FN)
    FPR = FP / (FP+TN)
    Accuracy = (TP + TN) * 1.0 /  (TP + TN + FP + FN)
    F_Score = 2*((Precision*Recall)/(Precision+Recall))
    print ("%s\t\t%.2f\t\t\t%.2f\t\t\t%.2f\t\t%.2f\t\t\t%.2f"%(name,Precision,Recall,FPR, Accuracy,F_Score, ))

def save_metrics_to_json():
    pass

''' 
save model to file for documentation purposes
filename is date, time and used machine learning algorithm
'''
def save_model_to_file(model):
    # get time_stamp
    now = datetime.now() # current date and time
    date = now.strftime('%d' + '-' + '%m' + '-' + '%Y' + '_' + '%H%M%S')

    file_name = date + '_' + model

    ''' Saving the model to file '''
    import joblib
 
    # Save the model as a pickle in a file
    joblib.dump(model, file_name)

# TODO: Eventuell zum Auffetten der Metriken
'''
accuracy_score 
'''
def print_metrics_to_cli(accuracy_score, y_test, predictions):
    print("Accuracy : %.2f"%(accuracy_score(y_test, predictions)*100))
    print("Confusion Matrix:\n",  confusion_matrix(y_test, predictions))
    print("Recall : %.2f"%(recall_score(y_test, predictions)*100))
    print("Precision : %.2f"%(precision_score(y_test, predictions)*100))
    print("F1 Score : %.2f"%(f1_score(y_test, predictions)*100))
    print("ROC : %.2f"%(roc_auc_score(y_test,predictions)*100))





