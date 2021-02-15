__all__ = ['getCurDir', 'getDatasetPath', 'showPlot', 'createEmailDataset', 'createDoubleMixupEmailDataset', 'shuffleDataset', 'cleanCache', 'printResult']

import fastbook
import codecs
import torch
import sys
import gc
import pandas as pd
import numpy as np

from fastbook import *
from multiprocessing import Pool

def getCurDir():
    sysPath = sys.path
    return sysPath[0]

def getDatasetPath(dir):
    return Path(dir + '/enron1')

def showPlot(names, values):
    plt.bar(names, values)
    plt.show()

'''
 Local function
'''
def loadEmail(filepath):
    with codecs.open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        raw_email = str(file.read())
        raw_email = raw_email.replace('\r\n', ' ')
        email = re.sub(r"[^a-zA-Z0-9]+", ' ', raw_email)
    return email

def createEmailDataset(filepaths, label):
    all_emails = L(loadEmail(filepath) for filepath in filepaths)
    
    df_emails = []
    
    for i in range(len(all_emails)):
        df_emails.append(all_emails[i])
    
    df_emails = pd.DataFrame(df_emails, columns = ['Content'])
    df_emails['Label'] = label
    
    return df_emails

def createDoubleMixupEmailDataset(filepaths, label, subFilepaths):
    all_emails = L(loadEmail(filepath) for filepath in filepaths)
    all_sub_emails = L(loadEmail(subFilepath) for subFilepath in subFilepaths)
    
    df_emails = []
    
    for i in range(len(all_emails)):
        df_emails.append(all_emails[i])
    
    for i in range(len(all_emails)):
        randSubIndex = np.random.randint(len(all_sub_emails) - 1)
        sub_email = all_sub_emails[randSubIndex]
        df_emails.append(sub_email[:len(sub_email)//2] + all_emails[i])
        
    df_emails = pd.DataFrame(df_emails, columns = ['Content'])
    df_emails['Label'] = label
    
    return df_emails
    
def shuffleDataset(all_dataframe):
    all_dataframe = all_dataframe.dropna()
    all_dataframe = all_dataframe.sample(frac = 1).reset_index(drop = True)
    
    return all_dataframe

def cleanCache():
    torch.cuda.empty_cache()
    gc.collect()

    print("Cuda available: ", torch.cuda.is_available())
#     print("Device: ", torch.cuda.get_device_name(0))
    print("Current GPU memory by tensors: ", torch.cuda.memory_allocated())
    print("Current GPU memory by caching: ", torch.cuda.memory_reserved())
    
def printResult(expected, output):
    print('Expect: ', expected)
    if (expected == int(output[1])):
        print('Predict result: (', output[1], ', ', output[2], ', ', output[3], ')')
    else:
        print('Fail to predict\nPredict result: ', output)
    print('------------------\n')
