import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import output

def run_art_evasion_pgd(X, y, t_size, SKmodel):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=4)
    SKmodel.fit(X_train, y_train)

    from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression
    estimator = ScikitlearnLogisticRegression(model=SKmodel, clip_values=(0,1))

    y_test = y_test.to_numpy()

    # Evaluate the ART classifier on benign test examples
    y_pred = estimator.predict(X_test)

    # takes the higher value of each prediction array and makes 
    y_pred = np.argmax(y_pred, axis=1)
    # get confusion matrix and print results
    matrix = confusion_matrix(y_test, y_pred)
    output.print_results_from_matrix('ART_Evasion_Benign', matrix)

    # Generate adversarial test examples
    from art.attacks.evasion import ProjectedGradientDescent
    attack = ProjectedGradientDescent(estimator=estimator, eps=0.3)

    x = X_test.toarray()
    x_test_adv = attack.generate(x)

    # Evaluate the ART classifier on adversarial test examples
    y_pred = estimator.predict(x_test_adv)
    # takes the higher value of each prediction array and makes 
    y_pred = np.argmax(y_pred, axis=1)

    matrix = confusion_matrix(y_test, y_pred)
    output.print_results_from_matrix('ART_Evasion_PGD', matrix)

def run_art_poisoning_svm(X, y, t_size, SKmodel):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=4)
    SKmodel.fit(X_train, y_train)

    from art.estimators.classification.scikitlearn import ScikitlearnLinearSVC
    estimator = ScikitlearnLinearSVC(model=SKmodel, clip_values=(0,1))

    X_train = X_train.toarray()
    #X_test = X_test.to_numpy()
    y_train = y_train.toarray()
    #y_test = y_test.to_numpy()

    # Evaluate the ART classifier on benign test examples
    y_pred = estimator.predict(X_test)

    # takes the higher value of each prediction array and makes 
    y_pred = np.argmax(y_pred, axis=1)
    # get confusion matrix and print results
    matrix = confusion_matrix(y_test, y_pred)
    output.print_results_from_matrix('ART_Poisoning_Benign', matrix)

    ########################
    '''
    x_val, y_val are optional if 
    '''
    # Generate adversarial test examples
    from art.attacks.poisoning import PoisoningAttackSVM
    attack = PoisoningAttackSVM(estimator=estimator, 
        eps=0.3, 
        x_train=X_train,
        y_train=y_train,
        x_val=X_test,
        y_val=y_test,
        max_iter=1000,
        verbose=True)

    # poison
    '''
        x_adv    ... poisoning examples
        y_attack ... poisoning labels
    '''
    x_adv, y_attack = attack.poison(X_train[6],y_train[6])

    # generate attack point

    final_attack_point, poinsoned_model = attack.generate_attack_point(x_attack=x_adv, y_attack=y_attack)    

    # Evaluate the ART classifier on adversarial test examples
    y_pred = poinsoned_model.predict(X_test)
    # takes the higher value of each prediction array and makes 
    y_pred = np.argmax(y_pred, axis=1)

    matrix = confusion_matrix(y_test, y_pred)
    output.print_results_from_matrix('ART_Poisoning_SVM', matrix)


def run_art_evasion_universal_perturbation(X, y, t_size, SKmodel):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=4)

    SKmodel.fit(X_train, y_train)

    from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier
    classifier = ScikitlearnRandomForestClassifier(model=SKmodel, clip_values=(0,1))

    y_test = y_test.to_numpy()

    # Evaluate the ART classifier on benign test examples

    y_pred = classifier.predict(X_test)

    # takes the higher value of each prediction array and makes 
    y_pred = np.argmax(y_pred, axis=1)
    

    matrix = confusion_matrix(y_test, y_pred)
    output.print_results_from_matrix('ART_Evasion_Benign', matrix)

    # Generate adversarial test examples
    from art.attacks.evasion import UniversalPerturbation
    attack = UniversalPerturbation(classifier=classifier, attacker='deepfool', verbose=True)

    x = X_test.toarray()
    x_test_adv = attack.generate(x)

    # Evaluate the ART classifier on adversarial test examples
    y_pred = classifier.predict(x_test_adv)
    # takes the higher value of each prediction array and makes 
    y_pred = np.argmax(y_pred, axis=1)

    matrix = confusion_matrix(y_test, y_pred)
    output.print_results_from_matrix('ART_Evasion_UniversalPerturbation', matrix)