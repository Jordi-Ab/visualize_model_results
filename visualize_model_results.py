import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def calc_metrics(y_score, y_real, vis=True):
    fpr, tpr, _ = roc_curve(y_real, y_score)
    roc_auc = auc(fpr, tpr)
    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr)).set_index('fpr')
    if vis:
        fig, ax = plt.subplots(ncols=3, figsize=(30, 10))
        plt.figure()
        lw = 2
        df.plot(color='darkorange', ax=ax[0], legend=False, grid=True)
        ax[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    
    d = pd.DataFrame({'y_score':np.squeeze(y_score), 'trues':y_real})
    # make predictions
    bins = np.arange(0, 1.01, 0.01).tolist()
    good_bad = d.groupby([pd.cut(d.y_score, bins), 'trues']).size().unstack()
    good_bad = good_bad.replace(np.nan, 0)
    good_bad.columns = ["BAD_CLIENT", "GOOD_CLIENT"]
    good_bad["TOTAL"] = good_bad["GOOD_CLIENT"] + good_bad["BAD_CLIENT"]
    good_bad["GOOD_RATE"] = good_bad["GOOD_CLIENT"] / (good_bad["GOOD_CLIENT"] + good_bad["BAD_CLIENT"])
    good_bad["BAD_RATE"] = good_bad["BAD_CLIENT"] / (good_bad["GOOD_CLIENT"] + good_bad["BAD_CLIENT"])
    good_bad["f(x)_GOOD_CLIENT"] = good_bad["GOOD_CLIENT"] / good_bad["GOOD_CLIENT"].sum()
    good_bad["f(x)_BAD_CLIENT"] = good_bad["BAD_CLIENT"] / good_bad["BAD_CLIENT"].sum()
    good_bad["F(x)_GOOD_CLIENT"] = good_bad["f(x)_GOOD_CLIENT"].cumsum()
    good_bad["F(x)_BAD_CLIENT"] = good_bad["f(x)_BAD_CLIENT"].cumsum()
    good_bad = good_bad.reset_index()
    ks = abs(good_bad['F(x)_GOOD_CLIENT'] - good_bad['F(x)_BAD_CLIENT'])
    if vis:
        ax[0].set_title('ROC curve (area = %0.2f)' % roc_auc)
        good_bad[['F(x)_GOOD_CLIENT','F(x)_BAD_CLIENT']].plot(ax=ax[1], title='KS metric: ' + str(round(ks.max(), 3)))
        ax[2].set_xlabel('predicted')
        ax[2].set_ylabel('frequency')
        ax[2].set_title('Population seperation')
        ax[2].hist(d.loc[d.trues==1].y_score, bins = 50, color='red', alpha=0.3)
        ax[2].hist(d.loc[d.trues==0].y_score, bins = 50, color = 'blue', alpha=0.3)
        plt.show()
    #print(roc_auc, ks.max())
    return roc_auc, ks.max()

def calculate_iv(var, metric):
    df = pd.concat([var, metric], axis=1)
    df.columns = ["var", "metric"]
    df = df.groupby(["var", "metric"]).size().unstack()
    df.columns = ["Good", "Bad"]
    df["Total"] = df["Bad"] + df["Good"]
    df["Br"] = df["Bad"] / df["Total"]
    df["f(x)Good"] = (df["Good"] / df["Good"].sum())
    df["f(x)Bad"] = (df["Bad"] / df["Bad"].sum())
    df = df.fillna(0.001)
    df["WOE"] = np.log(df["f(x)Good"] / df["f(x)Bad"])
    df["IV"] = df["WOE"] * (df["f(x)Good"] - df["f(x)Bad"])
    return df

def plot_pop_separation(y_score, y_real, ax=None):
    if not ax:
        fig = plt.figure(figsize=(7,7))
        ax = plt.subplot(111)

    d = pd.DataFrame({'y_score':np.squeeze(y_score), 'trues':y_real})

    ax.set_xlabel('predicted')
    ax.set_ylabel('frequency')
    ax.set_title('Population seperation')
    ax.hist(d.loc[d.trues==1].y_score, bins = 50, color='red', alpha=0.3)
    ax.hist(d.loc[d.trues==0].y_score, bins = 50, color = 'blue', alpha=0.3)

def plot_ks(y_score, y_real, ax=None):
    if not ax:
        fig = plt.figure(figsize=(7,7))
        ax = plt.subplot(111)

    d = pd.DataFrame({'y_score':np.squeeze(y_score), 'trues':y_real})
    # make predictions
    bins = np.arange(0, 1.01, 0.01).tolist()
    good_bad = d.groupby([pd.cut(d.y_score, bins), 'trues']).size().unstack()
    good_bad = good_bad.replace(np.nan, 0)
    good_bad.columns = ["BAD_CLIENT", "GOOD_CLIENT"]
    good_bad["TOTAL"] = good_bad["GOOD_CLIENT"] + good_bad["BAD_CLIENT"]
    good_bad["GOOD_RATE"] = good_bad["GOOD_CLIENT"] / (good_bad["GOOD_CLIENT"] + good_bad["BAD_CLIENT"])
    good_bad["BAD_RATE"] = good_bad["BAD_CLIENT"] / (good_bad["GOOD_CLIENT"] + good_bad["BAD_CLIENT"])
    good_bad["f(x)_GOOD_CLIENT"] = good_bad["GOOD_CLIENT"] / good_bad["GOOD_CLIENT"].sum()
    good_bad["f(x)_BAD_CLIENT"] = good_bad["BAD_CLIENT"] / good_bad["BAD_CLIENT"].sum()
    good_bad["F(x)_GOOD_CLIENT"] = good_bad["f(x)_GOOD_CLIENT"].cumsum()
    good_bad["F(x)_BAD_CLIENT"] = good_bad["f(x)_BAD_CLIENT"].cumsum()
    good_bad = good_bad.reset_index()
    ks = abs(good_bad['F(x)_GOOD_CLIENT'] - good_bad['F(x)_BAD_CLIENT'])

    good_bad[
        ['F(x)_GOOD_CLIENT','F(x)_BAD_CLIENT']
    ].plot(
        ax=ax, 
        title='KS metric: ' + str(round(ks.max(), 3))
    )

def plot_woe(df, vars):
    """
    Plots distribution Bad/Good over 10 bins
    Inputs:
        df: DataFrame with vars and metric, metric should be 0/1
        vars: variables to plot

    TO-DO: add missing as a category
    """
    for v in vars:
        print(v)
        cuts = [-np.inf] + list(df.sort_values(v)[v].values[::int(len(df)/10)])[1:-1] + [np.inf]
        df['binned'] = pd.cut(df[v], cuts, duplicates='drop')
        s0 = df.loc[df.metric==0]
        s1 = df.loc[df.metric==1]
        fig, ax = plt.subplots()
        ax.plot(((s1.groupby('binned').count() / len(s1)).metric.values) / (s0.groupby('binned').count() / len(s0)).metric.values)
        ax.axhspan(0.8, 1.2, color='r', alpha=0.4)
        plt.xticks(np.arange(len((s1.loc[s1.metric==1].groupby('binned').groups.keys()))), (s1.loc[s1.metric==1].groupby('binned').groups.keys()), rotation=20)
        plt.show()

def plot_roc(
    y_score, 
    y_real, 
    ax=None, 
    model_name='model', 
    print_auc=True,
    color=None
):

    fpr, tpr, thresholds = roc_curve(y_real, y_score)
    auc = roc_auc_score(y_real, y_score)

    if not ax:
        fig = plt.figure(figsize=(7,7))
        ax = plt.subplot(111)

    ax.plot(
        fpr, 
        tpr, 
        label=model_name + " AUC = {0:.2f}".format(auc),
        color=color
    )
    ax.plot(
        [0, 1], 
        [0, 1], 
        color='navy', 
        lw=2, 
        linestyle='--'
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")

    if print_auc:
        print('Area under ROC: '+str(auc))

def _chunks(l, n):
    step = int(len(l)/n)
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), step):
        yield l[i:i + step]

def _plot_prob(probas_array, ax, model_name, color):
    """
    probas_array must be a two dimensional numpy array,
    with the column1 being the frequency of real observations
    and column2 the average predicted proba for each chunk.
    """
    max_prob = np.max(probas_array)
    linewidth=2.3; fontsize=13
    if not ax:
        fig = plt.figure(figsize=(7,7))
        ax = plt.subplot(111)
    #plt.xticks(fontsize=fontsize-1)
    #plt.yticks(fontsize=fontsize-1)
    ax.plot(
        [0,max_prob], 
        [0,max_prob], 
        color='navy', 
        lw=2, 
        linestyle='--'
    )
    ax.plot(
        probas_array[:,1], 
        probas_array[:,0], 
        '-o', 
        linewidth = linewidth,
        label = model_name,
        color=color
    )
    ax.grid()
    ax.set_title('Predicted Proba v.s Frequency')
    ax.set_ylabel('frequency probability', fontsize=fontsize)
    ax.set_xlabel('mean predicted probability', fontsize=fontsize)
    ax.legend()

def plot_predicted_vs_real_frequencies(
    y_score, 
    y_real,
    n_chunks = 10,
    ax = None,
    model_name='model',
    color=None
):
    """
    Generates a two dimensional array of size (chunks, 2) 
    with column1 being the frequency of real observations 
    and column2 the average predicted proba for each chunk.
    """
    chnks=n_chunks
    prob_real = pd.DataFrame(
        np.transpose(
            [y_score, y_real]
        ),
        columns=['prob','real']
    )
    prob_real.sort_values(
        by='prob', 
        ascending=False, 
        inplace=True
    )
    chunks_test = list(_chunks(prob_real['real'],chnks))
    chunks_test_p = list(_chunks(prob_real['prob'],chnks))
    probabilidad = np.array([
        [
            (sum(chunks_test[i]==1)/len(chunks_test[i])), # mean of real probas of chunk i
            np.mean(chunks_test_p[i]) # mean of predicted observations of chunk i
        ] for i in range(0,chnks)
    ]) # 2-D Array of mean real probas and mean of predicted probas
    _plot_prob(probabilidad, ax=ax, model_name=model_name, color=color)

def _plot_lift_at_churn(lifts, model_name, ax, plot_random):
    
    chnks = len(lifts[0])
    random_lift_at_chunk = np.array([[(i+1)*100/chnks, 1] for i in range(0,chnks)])
    
    linewidth=2.3
    fontsize=13
    if not ax:
        fig = plt.figure(figsize=(8,5))
        ax = plt.subplot(111)
    #plt.xticks(fontsize=fontsize-1)
    #plt.yticks(fontsize=fontsize-1)
    if plot_random:
        ax.plot(
            random_lift_at_chunk[:,0], 
            random_lift_at_chunk[:,1], 
            '--', 
            label='random', 
            linewidth = linewidth, 
            color='r'
        )
    for i in range(len(lifts)):
        ax.plot(
            lifts[i][:,0], 
            lifts[i][:,1], 
            '-o', 
            label=model_name, 
            linewidth = linewidth,
            color='b'
        )
    ax.grid()
    ax.legend(loc='upper right', prop={'size': fontsize})
    ax.set_xlabel('percentile', fontsize=fontsize)
    ax.set_ylabel('lift @ chunk', fontsize=fontsize)
    
def _plot_lift(lifts, model_name, ax, plot_ones, color):
    linewidth=2.3
    fontsize=13
    if not ax:
        fig = plt.figure(figsize=(8,5))
        ax = plt.subplot(111)
    #plt.xticks(fontsize=fontsize-1)
    #plt.yticks(fontsize=fontsize-1)
    for i in range(len(lifts)):
        ax.plot(
            lifts[i][:,0], 
            lifts[i][:,1], 
            '-o', 
            label=model_name, 
            linewidth = linewidth,
            color=color
        )
        if plot_ones:
            ax.plot(
                lifts[i][:,0], 
                np.ones(len(lifts[i][:,1])), 
                'r--', 
                label='ones', 
                linewidth = linewidth
            )
    ax.grid()
    ax.legend(loc='upper right', prop={'size': fontsize})
    ax.set_xlabel('percentile', fontsize=fontsize)
    ax.set_ylabel('lift', fontsize=fontsize)
        
def plot_lifts(
    y_score,
    y_real,
    model_name='model', 
    chnks=10, 
    ax=None, 
    plot_ones=True,
    color=None
):
    
    lifts = [_compute_lift(y_score, y_real, chnks, at_churn=False)]
    _plot_lift(lifts, model_name, ax, plot_ones, color)

def plot_lifts_at_churn(
    y_score,
    y_real,
    model_name='model', 
    chnks=10, 
    ax=None, 
    plot_random=True
):
    
    lifts = [_compute_lift(y_score, y_real, chnks, at_churn=True)]
    _plot_lift_at_churn(lifts, model_name, ax, plot_random)
    
def _compute_lift(y_score, y_real, chnks, at_churn=False):
    #Define an auxiliar dataframe to plot the curve
    aux_lift = pd.DataFrame()
    #Create a real and predicted column for our new DataFrame and assign values
    aux_lift['real'] = y_real
    aux_lift['predicted'] = y_score
    #Order the values for the predicted probability column:
    aux_lift.sort_values('predicted',ascending=False,inplace=True)
    chunks_test = list(_chunks(aux_lift['real'],chnks))
    
    if at_churn:
        prior_rate = np.mean(pd.concat(chunks_test[:chnks])==1)
        lift = np.array(
            [
                [
                    (i+1)*100/chnks, 
                    (np.mean(chunks_test[i]==1)/prior_rate) 
                ] for i in range(0,chnks)
            ]
        )
    else:
        all_cases = sum(pd.concat(chunks_test[:chnks])==1)
        lift = np.array(
            [
                [
                    (i+1)*100/chnks, 
                    (sum(pd.concat(chunks_test[:i+1])==1)/all_cases)/((i+1)/chnks) 
                ] for i in range(0,chnks)
            ]
        )
    return lift