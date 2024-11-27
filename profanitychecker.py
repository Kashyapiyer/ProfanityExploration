import logging 
logging.getLogger().setLevel(logging.INFO)
from better_profanity import profanity
import warnings
import pandas as pd 
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', None)


def evaluatesentenceprofanity(contextstr,censor=False):
    try: 
        evaldict = {}
        result = profanity.contains_profanity(contextstr)
        if (result == True) and censor==True:
          censored_text = profanity.censor(contextstr)
          evaldict['contextpassed'] = contextstr 
          evaldict['censored_context'] = censored_text
          evaldict['has_profanity'] = 0
          return evaldict
        
        elif (result == True) and censor==False:
          evaldict['contextpassed'] = contextstr 
          evaldict['has_profanity'] = 1
          logging.error('contains profanity')
          return evaldict

        else:
          evaldict['contextpassed'] = contextstr
          evaldict['has_profanity'] = 0
          evaldict['censored_context'] = None
          logging.info('does not contain profanity')
        return evaldict
      
    except Exception as e: 
      return f'Encountered error while checking profanity on supplied context{e}'



def calculateprofanity(df, contextstrcol,gtlabel='toxicity',censor=False):
    try: 
        if not censor: 
            df['profanity_result'] = df[contextstrcol].apply(lambda t: profanity.contains_profanity(t))
            df['censored_text'] = None
            
        else: 
            df['profanity_result'] = df[contextstrcol].apply(lambda t: profanity.contains_profanity(t))
            df['censored_text'] = df[contextstrcol].apply(lambda x: profanity.censor(x))
            
        df['has_profanity'] = df['profanity_result'].apply(lambda x: 1 if x else 0)
           
        df['match'] = df[gtlabel] == df['has_profanity']
        return df
    except Exception as e: 
        return f'Encountered error while checking profanity on supplied context{e}'


def benchmark_profanity(df, contextstrcol, exportfilepath,gtlabel='toxicity',censor=False):
    from datetime import datetime 
    starttime= datetime.now()
    result = calculateprofanity(df,contextstrcol,gtlabel,censor)
    result.to_csv(exportfilepath, index=False)
    endtime = datetime.now()
    durationmins = ((endtime - starttime).total_seconds()) / 60
    logging.info(f'Latency duration in mins : {durationmins}')
    return result

def calculate_metrics(df,truecol,predcol):
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    y_true = df[truecol]
    y_pred = df[predcol]	

    true_negative, false_positive, false_negative, true_postive = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result = {}
    result['Accuracy'] = f'{accuracy_score(y_true, y_pred):.4f}'
    result['Precision']= f'{precision_score(y_true, y_pred):.4f}'
    result['Recall'] = f'{recall_score(y_true, y_pred):.4f}'
    result['F1_score_cal'] = f'{f1_score(y_true, y_pred):.4f}'
    result['true_negative'] = true_negative
    result['true_postive'] = true_postive
    result['false_negative'] = false_negative
    result['false_positive'] = false_positive
    return result
