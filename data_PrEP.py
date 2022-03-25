import random
import pandas as pd
from sklearn.utils import resample

from torch import absolute

def create_dataframe_from_txt(textfile):
    df = pd.read_table(textfile, names=['request', 'label'])
    return df

def create_dataframe_from_sample(sample):
    df = pd.DataFrame(sample, columns=['request'])
    df['label'] = ' '
    return df

def assign_label_to_dfcolumn_label(dataframe, label):
    dataframe = dataframe.assign(label=label)
    return dataframe

def get_random_sample(filename, n_samples):
    with open(filename) as f:
        r_sample = f.read().splitlines()
    r_sample = random.sample(r_sample, n_samples)
    return r_sample

#
def create_injected_dataframe(dataframe1, dataframe2, size):
    if len(dataframe1) >= size:
        dataframe1_downsampled = resample(dataframe1, replace=True, n_samples=(size)-(len(dataframe2)), random_state=100)
        inj_df = pd.concat([dataframe1_downsampled,dataframe2])
    else:
        inj_df = pd.concat([dataframe1,dataframe2])
        inj_df_upsampled = resample(inj_df, replace=True, n_samples=size, random_state=100)
        inj_df = inj_df_upsampled
    return inj_df

# TODO: what if both dataframes are < 36000
def create_balanced_dataframe(dataframe1, dataframe2, size):
    if len(dataframe1) != size/2:
        dataframe1 = resample(dataframe1, replace=True, n_samples=(int(size/2)), random_state=100)
    elif len(dataframe2) != size/2:
        dataframe2 = resample(dataframe2, replace=True, n_samples=(int(size/2)), random_state=100)
    else:
        pass
    balanced_df = pd.concat([dataframe1,dataframe2])
    return balanced_df

def create_n_size_dataframe(dataframe, size):
    df = resample(dataframe, replace=True, n_samples=size, random_state=100)
    return df


def create_labelled_dataframe(filename, label):
    df = pd.read_table(filename, names=['request', 'label'])
    df = df.assign(label=label)
    return df

# TODO: this needs redo, so it can handle a list of files, or list of dict --> filename + label
def create_full_labelled_dataframe():
    df_0 = create_labelled_dataframe('normalTrafficTraining.txt', 0)
    df_1 = create_labelled_dataframe('anomalousTrafficTest.txt', 1)
    df = pd.concat([df_0,df_1])
    return df

def add_adversarial_data(dataframe, label):
    with open('/Users/jan/Documents/GitHub/ma-thesis_cs20m019/adversarial.txt') as f:
        all_log4j = f.read().splitlines()
    random_log4j = random.sample(all_log4j, 100)
    df_log4j = pd.DataFrame(random_log4j, columns=['request'])
    df_log4j['label'] = label
    dataframe = pd.concat([dataframe, df_log4j])
    return dataframe

def resample_malicious_dataset(dataframe, n_samples, random_state):
    df = dataframe
    # divide in normal and malicious events
    df_0 = df[df.threat==0]
    df_1 = df[df.threat==1]
    print('Resampling malicous sample up to [', + n_samples + '] samples')
    # creates additional samples up to n_samples
    df_1_up = resample(df_1,replace=True,n_samples=n_samples,random_state=random_state)
    df = pd.concat([df_0,df_1_up])
    return df
