import sys
import pandas as pd
import data_PrEP as DP
import model as MOD
import output as OUT
import adversarial as ADV
import usecase_art as UCART
#TODO: remove this


# dataset files
# normal requests 
normal_requests = '/home/jan/master_thesis/ma-thesis_cs20m019/datasets/normal_requests.txt'
# anomalous requests
anomalous_requests = '/home/jan/master_thesis/ma-thesis_cs20m019/datasets/anomalous_requests.txt'
#log4j requests
log4j_requests = '/home/jan/master_thesis/ma-thesis_cs20m019/datasets/log4j_requests.txt'
# full dataset size
full_size = 72000
test_size = 0.2

# benchmark for csic 2010 dataset
def run_scenario1():
    # create full-sized dataframe
    df_0 = DP.create_dataframe_from_txt(normal_requests)
    df_1 = DP.create_dataframe_from_txt(anomalous_requests)
    df_0 = DP.assign_label_to_dfcolumn_label(df_0, 0)
    df_1 = DP.assign_label_to_dfcolumn_label(df_1, 1)
    df = DP.create_balanced_dataframe(df_0, df_1, full_size)

    X = MOD.get_observed_data(df)
    y = MOD.get_label(df)

    #X = MOD.vectorize_observed_data(X)
    X = MOD.count_vectorize_observed_data(X)
    MOD.run_all_models(X, y, test_size)

# benchmark für csic 2010 dataset with new log4j data
def run_scenario2():
    # create full-sized dataframe
    df_0 = DP.create_dataframe_from_txt(normal_requests)
    df_1 = DP.create_dataframe_from_txt(anomalous_requests)
    df_l4j = DP.create_dataframe_from_sample(DP.get_random_sample(log4j_requests, 100))
    df_0 = DP.assign_label_to_dfcolumn_label(df_0, 0)
    df_1 = DP.assign_label_to_dfcolumn_label(df_1, 1)
    df_l4j = DP.assign_label_to_dfcolumn_label(df_l4j, 1)

    # add labelled log4j requests to anomalous requests
    df_1 = pd.concat([df_1, df_l4j])
    # create a completely labelled dataset from normal and anomalous requests
    df = DP.create_balanced_dataframe(df_0, df_1, full_size)
    # split into data and labels
    X = MOD.get_observed_data(df)
    y = MOD.get_label(df)
    X = MOD.count_vectorize_observed_data(X)

    MOD.run_all_models(X, y, test_size)

# run ART evasion attack PGD on regression algorithms
def run_scenario3():
    # create full-sized dataframe
    df_0 = DP.create_dataframe_from_txt(normal_requests)
    df_1 = DP.create_dataframe_from_txt(anomalous_requests)
    df_l4j = DP.create_dataframe_from_sample(DP.get_random_sample(log4j_requests, 100))
    df_0 = DP.assign_label_to_dfcolumn_label(df_0, 0)
    df_1 = DP.assign_label_to_dfcolumn_label(df_1, 1)
    df_l4j = DP.assign_label_to_dfcolumn_label(df_l4j, 1)

    # add labelled log4j requests to anomalous requests
    df_1 = pd.concat([df_1, df_l4j])
    # create a completely labelled dataset from normal and anomalous requests
    df = DP.create_balanced_dataframe(df_0, df_1, full_size)
    # split into data and labels
    X = MOD.get_observed_data(df)
    y = MOD.get_label(df)
    X = MOD.count_vectorize_observed_data(X)

    # Projected Gradient Descent (PGD)
    model = MOD.get_SKLearn_model('LogisticRegression')
    UCART.run_art_evasion_pgd(X, y, test_size, model)


# run ART poisoning attack SVM attack on SVC
def run_scenario4():
    # create full-sized dataframe
    df_0 = DP.create_dataframe_from_txt(normal_requests)
    df_1 = DP.create_dataframe_from_txt(anomalous_requests)
    df_l4j = DP.create_dataframe_from_sample(DP.get_random_sample(log4j_requests, 100))
    df_0 = DP.assign_label_to_dfcolumn_label(df_0, 0)
    df_1 = DP.assign_label_to_dfcolumn_label(df_1, 1)
    df_l4j = DP.assign_label_to_dfcolumn_label(df_l4j, 1)

    # add labelled log4j requests to anomalous requests
    df_1 = pd.concat([df_1, df_l4j])
    # create a completely labelled dataset from normal and anomalous requests
    df = DP.create_balanced_dataframe(df_0, df_1, 2000)
    # split into data and labels
    X = MOD.get_observed_data(df)
    y = MOD.get_label(df)
    X = MOD.count_vectorize_observed_data(X)

    # poisoning attack on Support Vector Machines (SVM) by Biggio
    model = MOD.get_SKLearn_model('LinearSVClassifier')
    UCART.run_art_poisoning_svm(X, y, test_size, model)

# run ART evasion universal perturbation attack on random forest classifier
def run_scenario5():
    # create full 72k dataframe
    df_0 = DP.create_dataframe_from_txt(normal_requests)
    df_1 = DP.create_dataframe_from_txt(anomalous_requests)
    df_0 = DP.assign_label_to_dfcolumn_label(df_0, 0)
    df_1 = DP.assign_label_to_dfcolumn_label(df_1, 1)
    df = DP.create_balanced_dataframe(df_0, df_1, 5000)

    X = MOD.get_observed_data(df)
    y = MOD.get_label(df)

    X = MOD.count_vectorize_observed_data(X)

    model = MOD.get_SKLearn_model('RandomForestClassifier')

    UCART.run_art_evasion_universal_perturbation(X, y, test_size, model)

# run evasion attack Carlini & Wagner L2, against MLPC?
def run_scenario6(): 
    pass


def main():
    
    run_scenario1()
    #run_scenario2()
    #run_scenario3()
    #run_scenario4()
    #run_scenario5()
    #run_scenario6()


if __name__ == "__main__":
    main()