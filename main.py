import sys
import pandas as pd
import data_PrEP as DP
import model as MOD
import output as OUT
import adversarial as ADV
#TODO: remove this


# dataset files
# normal requests 
normal_requests = '/Users/jan/Documents/GitHub/ma-thesis_cs20m019/datasets/normal_requests.txt'
# anomalous requests
anomalous_requests = '/Users/jan/Documents/GitHub/ma-thesis_cs20m019/datasets/anomalous_requests.txt'
#log4j requests
log4j_requests = '/Users/jan/Documents/GitHub/ma-thesis_cs20m019/datasets/log4j_requests.txt'
# full dataset size
full_size = 72000

# benchmark for csic 2010 dataset
def run_scenario1():
    # create full 72k dataframe
    df_0 = DP.create_dataframe_from_txt(normal_requests)
    df_1 = DP.create_dataframe_from_txt(anomalous_requests)
    df_0 = DP.assign_label_to_dfcolumn_label(df_0, 0)
    df_1 = DP.assign_label_to_dfcolumn_label(df_1, 1)
    df = DP.create_balanced_dataframe(df_0, df_1, full_size)

    X = MOD.get_observed_data(df)
    y = MOD.get_label(df)

    #X = MOD.vectorize_observed_data(X)
    X = MOD.count_vectorize_observed_data(X)
    MOD.run_all_models(X, y, 0.2)

# run log4j-testset against benchmark models
def run_scenario2():
    # create full 72k dataframe
    df_0 = DP.create_dataframe_from_txt(normal_requests)
    df_1 = DP.create_dataframe_from_txt(anomalous_requests)
    df_0 = DP.assign_label_to_dfcolumn_label(df_0, 0)
    df_1 = DP.assign_label_to_dfcolumn_label(df_1, 1)
    df = DP.create_balanced_dataframe(df_0, df_1, full_size)

    X = MOD.get_observed_data(df)
    y = MOD.get_label(df)

    X = MOD.vectorize_observed_data(X)

    print('Shape of first X after vectorize: ', X.shape)

    scen1_model = MOD.get_trained_model('rfc', X, y, 0.2)

    # create df with correctly labelled data for training set
    df_0 = DP.create_dataframe_from_txt(normal_requests)
    df_log4j = DP.create_dataframe_from_txt(log4j_requests)
    df_0 = DP.assign_label_to_dfcolumn_label(df_0, 0)
    df_log4j = DP.assign_label_to_dfcolumn_label(df_log4j, 1)
    df = DP.create_balanced_dataframe(df_0, df_log4j, full_size)

    X = MOD.get_observed_data(df)
    y = MOD.get_label(df)

    X = MOD.vectorize_observed_data(X)

    print('Shape of second X after vectorize: ', X.shape)

    df_X_train, df_X_test, df_y_train, df_y_test = MOD.get_set_matrix(X, y, 0.2)

    matrix = MOD.predict_with_existing_trained_model(scen1_model, df_X_test, df_y_test)

    if matrix == None:
        print('Error due to .s..s.')
    else:
        OUT.print_results_from_matrix(matrix, 'rfc')

# run with log4j-dataset in both training and testset
def run_scenario3():
    df_0 = DP.create_dataframe_from_txt(normal_requests)
    df_1 = DP.create_dataframe_from_txt(anomalous_requests)
    df_log4j = DP.create_dataframe_from_sample(DP.get_random_sample(log4j_requests, 100))
    df_1 = DP.create_injected_dataframe(df_1, df_log4j, int(full_size/2))

    df_0 = DP.assign_label_to_dfcolumn_label(df_0, 0)
    df_1 = DP.assign_label_to_dfcolumn_label(df_1, 1)

    df = DP.create_balanced_dataframe(df_0, df_1, full_size)

    X = MOD.get_observed_data(df)
    y = MOD.get_label(df)

    X = MOD.vectorize_observed_data(X)
    
    matrix = MOD.run_model('rfc', X, y, 0.2)
    OUT.print_results_from_matrix(matrix, 'Random Forest(tree=50)')

# run with poisoned log4j-dataset
def run_scenario4():
    df_0 = DP.create_dataframe_from_txt(normal_requests)
    df_1 = DP.create_dataframe_from_txt(anomalous_requests)

    df_log4j = DP.create_dataframe_from_sample(DP.get_random_sample(log4j_requests, 14000))
    #df_log4j2 = DP.create_dataframe_from_sample(DP.get_random_sample(log4j_requests, 100))

    df_0 = DP.create_injected_dataframe(df_0, df_log4j, int(full_size/2))
    #df_1 = DP.create_injected_dataframe(df_1, df_log4j2, int(full_size/2))

    df_0 = DP.assign_label_to_dfcolumn_label(df_0, 0)
    df_1 = DP.assign_label_to_dfcolumn_label(df_1, 1)

    df = DP.create_balanced_dataframe(df_0, df_1, full_size)

    X = MOD.get_observed_data(df)
    y = MOD.get_label(df)

    X = MOD.vectorize_observed_data(X)
    
    matrix = MOD.run_model('rfc', X, y, 0.2)
    OUT.print_results_from_matrix(matrix, 'Random Forest(tree=50)')

# run evasion attack
def run_scenario5(): 
    df_0 = DP.create_dataframe_from_txt(normal_requests)
    df_1 = DP.create_dataframe_from_txt(anomalous_requests)
    df_log4j = DP.create_dataframe_from_sample(DP.get_random_sample(log4j_requests, 100))
    df_1 = DP.create_injected_dataframe(df_1, df_log4j, int(full_size/2))

    df_0 = DP.assign_label_to_dfcolumn_label(df_0, 0)
    df_1 = DP.assign_label_to_dfcolumn_label(df_1, 1)

    df = DP.create_balanced_dataframe(df_0, df_1, full_size)

    X = MOD.get_observed_data(df)
    y = MOD.get_label(df)

    X = MOD.count_vectorize_observed_data(X)

    #X1 = MOD.vectorize_observed_data(X)
    
    # MOD.run_all_models(X1, y, 0.2)

    model = MOD.get_trained_model('dtc', X, y, 0.2) 
    #ADV.perfom_evasion_attack(X, y, random_state=100, clf=model, t_size=0.2)
    ADV.perform_evasion_cleverhans(X, y, random_state=100, clf=model, t_size=0.2)

# basically scenario3() with max_features when vectorizing
def run_scenario6():
    # create full 72k dataframe
    df_0 = DP.create_dataframe_from_txt(normal_requests)
    df_1 = DP.create_dataframe_from_txt(anomalous_requests)
    df_0 = DP.assign_label_to_dfcolumn_label(df_0, 0)
    df_1 = DP.assign_label_to_dfcolumn_label(df_1, 1)
    df = DP.create_balanced_dataframe(df_0, df_1, full_size)

    X = MOD.get_observed_data(df)
    y = MOD.get_label(df)

    X = MOD.vectorize_observed_data_with_maxfeatures(X, 21260)

    print('Shape of first X after vectorize: ', X.shape)

    scen1_model = MOD.get_trained_model('rfc', X, y, 0.2)

    # create df with correctly labelled data for training set
    df_0 = DP.create_dataframe_from_txt(normal_requests)
    df_log4j = DP.create_dataframe_from_txt(log4j_requests)
    df_0 = DP.assign_label_to_dfcolumn_label(df_0, 0)
    df_log4j = DP.assign_label_to_dfcolumn_label(df_log4j, 1)
    df = DP.create_balanced_dataframe(df_0, df_log4j, full_size)

    X = MOD.get_observed_data(df)
    y = MOD.get_label(df)

    X = MOD.vectorize_observed_data(X)

    print('Shape of second X after vectorize: ', X.shape)

    df_X_train, df_X_test, df_y_train, df_y_test = MOD.get_set_matrix(X, y, 0.2)

    matrix = MOD.predict_with_existing_trained_model(scen1_model, df_X_test, df_y_test)

    OUT.print_results_from_matrix(matrix, 'rfc')

def main():
    '''
    1. get parameters                                       --> helpers.py
    2. load files                                           --> data_PrEP.py
    3. prepare files                                        --> data_PrEP.py
    4. prepare ml-models accoring to set scenario           --> model.py
    5. create output: cli, json, model to file, graphic(s)  --> output.py
    '''

    #run_scenario1()
    #run_scenario2()
    #run_scenario3()
    #run_scenario4()
    run_scenario5()
    #run_scenario6()


if __name__ == "__main__":
    main()