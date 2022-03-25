import random
from tkinter.font import names
from turtle import clear
from venv import create
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing as pp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer


def create_labelled_dataframe(filename, label):
    df = pd.read_table(filename, names=['request', 'label'])
    df = df.assign(label=label)
    return df

def drop_label(dataframe, label):
    X = dataframe.drop(label, axis=1)
    return X

def print_stats_for_sparse_matrix(M, matrix_name):
    nnz = M.nnz
    index_ptr_array = M.indptr
    index_array = M.indices
    data_array = M.data
    non_zeros = M.count_nonzero


    print(matrix_name)
    print('NNZ of ', matrix_name)
    print(nnz)
    print('Number of Non-Zeroes in ', matrix_name)
    print(non_zeros)
    print('Index Pointer Array of ', matrix_name)
    print(index_ptr_array)
    print('Index Array of ', matrix_name)
    print(index_array)
    print('Data Array of ', matrix_name)
    print(data_array)


def show_sparse_matrix(csr_matrix):
    plt.spy(csr_matrix, markersize=1)
    plt.show()

normal_file = '/Users/jan/Documents/GitHub/ma-thesis_cs20m019/datasets/normal_requests.txt'
malicious_file = '/Users/jan/Documents/GitHub/ma-thesis_cs20m019/datasets/anomalous_requests.txt'
usecase_file = '/Users/jan/Documents/GitHub/ma-thesis_cs20m019/datasets/log4j_requests.txt'

vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char",ngram_range=(4, 4))

df_list = list()
df_0 = create_labelled_dataframe(normal_file, 0)
df_1 = create_labelled_dataframe(malicious_file, 1)
df_1_log = create_labelled_dataframe(usecase_file, 1)
df_list.append(df_0)
df_list.append(df_1)
df_list.append(df_1_log)

df_0 = drop_label(df_0, 'label')
df_1 = drop_label(df_1, 'label')
df_1_log = drop_label(df_1_log, 'label')

df_0 = vectorizer.fit_transform(df_0['request'])
df_1 = vectorizer.fit_transform(df_1['request'])
df_1_log = vectorizer.fit_transform(df_1_log['request'])

print_stats_for_sparse_matrix(df_0, 'Normal Matrix')
print_stats_for_sparse_matrix(df_1, 'Anomalous Matrix')
print_stats_for_sparse_matrix(df_1_log, 'Log4j Matrix')

show_sparse_matrix(df_1_log)


exit()

df_1 = resample(df_1,replace=True,n_samples=36000,random_state=100)

df = pd.concat([df_0,df_1])

##### plot sparsity matrix
import output
output.show_sparse_matrix(x)

print(x.shape)
exit()

