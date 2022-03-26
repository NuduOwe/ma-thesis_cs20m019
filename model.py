from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import output
'''
model_list = [
    'LGR',
    'SVC',
    'DTC',
    'RFC',
    'KNNC',
    'MLPC',
]
'''

model_list = [
    'LGR',
    'SVC',
    'DTC',
    'RFC',
]

# split dataframe into columns
def get_observed_data(df):
    x = df.drop("label",axis = 1)
    return x

def get_label(df):
     y = df.label
     return y

# feature extraction 

def count_vectorize_observed_data(observed_data):
    cnt_vectorizer = CountVectorizer(min_df=0.0, analyzer="char", ngram_range=(3, 3))
    X = cnt_vectorizer.fit_transform(observed_data['request'])
    return X

def vectorize_observed_data(observed_data):
    vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", ngram_range=(4, 4))
    X = vectorizer.fit_transform(observed_data['request'])
    return X

def vectorize_observed_data_with_maxfeatures(observed_data, max_features):
    vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", max_features=max_features, sublinear_tf=True, ngram_range=(2, 2))
    X = vectorizer.fit_transform(observed_data['request'])
    return X


def get_set_matrix(observed_data, label_data, test_size):
    set_matrix = train_test_split(observed_data, label_data, test_size=test_size, random_state=4)
    return set_matrix   

# run model functions
def run_all_models(observed_data, label_data, test_size):
    # split observed and label data into testset and trainingset
    X_train, X_test, y_train, y_test = train_test_split(observed_data, label_data, test_size=test_size, random_state=4)

    output.print_table_header()
    for i in model_list:
        if i == 'LGR':
            model = LogisticRegression(solver='lbfgs', max_iter=1000)
        elif i == 'SVC':
            model = LinearSVC(max_iter=1000)
        elif i == 'DTC':
            model = DecisionTreeClassifier()
        elif i == 'RFC':
            model = RandomForestClassifier(n_estimators=50)
        elif i == 'KNNC':
            model = KNeighborsClassifier()
        elif i == 'MLPC':
            model = MLPClassifier()
        else:
            pass
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred)
        output.print_results_from_matrix(i, matrix)


# TODO: Decide what functions to keep here

def get_SKLearn_model(model_name):
    if model_name == 'LogisticRegression':
        model = LogisticRegression(solver='lbfgs', max_iter=1000)
    elif model_name == 'LinearSVClassifier':
        model = LinearSVC(max_iter=1000)
    elif model_name == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier()
    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(n_estimators=50)
    elif model_name == 'KNearestNeighborClassifier':
        model = KNeighborsClassifier()
    elif model_name == 'MultiLayerPerceptronClassifier':
        model = MLPClassifier()
    else:
        pass
    return model


# run machine learning model with defined test_size
def run_model(model_name, observed_data, label_data, test_size):
    X_train, X_test, y_train, y_test = train_test_split(observed_data, label_data, test_size=test_size, random_state=4)

    if model_name == 'LogisticRegression':
        model = LogisticRegression()
    elif model_name == 'LinearSVClassifier':
        model = LinearSVC()
    elif model_name == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier()
    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(n_estimators=50)
    elif model_name == 'KNearestNeighborClassifier':
        model = KNeighborsClassifier()
    elif model_name == 'MultiLayerPerceptronClassifier':
        model = MLPClassifier()
    else:
        pass
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    return matrix


def run_model_with_split_sets(model_name, X_train, y_train, X_test, y_test):
    if model_name == 'rfc':
        model = RandomForestClassifier(n_estimators=50)
    elif model_name == 'dtc':
        model = DecisionTreeClassifier()
    elif model_name == 'MLPc':
        model = MLPClassifier()
    elif model_name == 'KNNc':
        model = KNeighborsClassifier()
    else:
        pass
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    return matrix

def get_trained_model(model_name, observed_data, label_data, test_size):
    '''shuffle=True for randomization of data before splitting'''
    X_train, X_test, y_train, y_test = train_test_split(observed_data, label_data, test_size=test_size, random_state=4)
    if model_name == 'rfc':
        model = RandomForestClassifier(n_estimators=50)
    elif model_name == 'dtc':
        model = DecisionTreeClassifier()
    elif model_name == 'MLPc':
        model = MLPClassifier()
    elif model_name == 'KNNc':
        model = KNeighborsClassifier()
    else:
        pass
    model.fit(X_train, y_train)

    return model

def predict_with_existing_trained_model(et_model, X_test, y_test):
    try:
        y_pred = et_model.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred)
        return matrix
    except:
        print('Cannot perfom prediction as number of features in X_train of model and X_test differ')


