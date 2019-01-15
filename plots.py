# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 17:49:11 2018

@author: Mathieu
"""

import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt

def scatter_plot_features(df,name='test',f1="Age",f2="rh_supramarginal_volume", other=None):
    """
    Plots two features as a scatterplot
    """
    sns.set_style("whitegrid")
    temp_df=pd.concat([df[f1],df[f2],df[other]],axis=1)
    plot = sns.scatterplot(data=temp_df,x=f1,y=f2,hue=other).get_figure()
    plot.savefig(name+'/'+'sp_'+f2+'_'+f1+'.png')
    plt.clf()

def dist_plot_features(df,name='test',f1="Age",f2="rh_supramarginal_volume", 
                       other=None):
    """
    Plots one features as a distplot
    """
    if f1==other:
        return False
    sns.set_style("whitegrid")
    print(f1)
#    temp_df=pd.concat([df[f1],df[f2],df[other]],axis=1)
    temp_serie=pd.Series(df[f1].copy(), name=str(f1))
    plot = sns.distplot(temp_serie).get_figure()
    plot.savefig(name+'_'+'dp_'+str(f1)+'.png')
    plt.clf()

def box_plot_features(df,name='test',f1="Age",f2="rh_supramarginal_volume", 
                       other=None):
    """
    Plots one feature as a box_plot. Use 'other' to separate between groups
    """
    if f1==other:
        return False
    
    plt.title(('Repartition of '+str(f1)))
    sns.set_style("whitegrid")
    temp_df=pd.concat([df[f1],df[other]],axis=1)
    plot = sns.boxplot(data=temp_df,x=f1,hue=other,palette="Set3")
    plot = sns.boxplot(data=temp_df,x=f1,hue=other,palette="Set3").get_figure()
    plot.savefig(name+'_'+'sbp_'+str(f1)+'.png')
    plt.clf()

lp = lambda i: plt.plot([],color=sc.cmap(sc.norm(i)), ms=np.sqrt(size), mec="none",
                        label="Feature {:g}".format(i), ls="", marker="o")[0]

def box_plot_features_cd(df,name='test',f1="Age",f2="rh_supramarginal_volume", 
                       other=None):
    """
    Plots one feature as a box_plot. Use 'other' to separate between groups
    """
    if f1==other:
        return False
    fig = plt.figure()
    sns.set_style("whitegrid")
    df_dict=dict()
    for dif in df[other].unique():
        df_dict[dif]=df.loc[df[other]==dif]
    
    for counter, dif in enumerate(df[other].unique()):
        plt.subplot(2, 1, counter+1) 
        if counter ==0 :plt.title(('Repartition of the '+str(f1)+' among patients'))
        sns.boxplot(data=df_dict[dif],
                           x=f1,
                           color =sns.color_palette("Paired", 8)[counter]).get_figure()
        plt.legend(labels=[dif],loc=4)

    plt.tight_layout()    
    plt.savefig(name+'_'+'sbp_with_hue_'+str(f1)+'.png')
    plt.show()
    plt.clf()
    plt.close('all')

def a_sl_of_plot(df_o,f1='Age',directory='plot_test',name='plots', target = 'ds', 
                 plot_function=dist_plot_features,other=None):
    df=df_o.copy()
    if not os.path.exists(directory):
        os.makedirs(directory)
    name = directory +'/'+ name
    df.describe().to_csv(str(name+'describe.csv'))
    for feature in list(df.columns):
#        try :   plot_function(df,name=name, f1=f1,f2=feature,other='ds')
#        except TypeError : print('Error for : ', feature)
        
        plot_function(df,name=name, f1=feature,f2=feature,other=other)
        
