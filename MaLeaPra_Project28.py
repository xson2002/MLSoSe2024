# -*- coding: utf-8 -*-
"""
Created on Sun May  5 12:46:16 2024

@author: leonz
"""

from spacy.lang.en import STOP_WORDS
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings
from hdbscan import HDBSCAN
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from functools import partial
import json
import os
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import torch

from datetime import datetime

'''
Folder Structure

    fit-model.py
    
    Dataset-Name-A:
      |
      |-dataset:
      |  |   
      |  |-item_metadata.csv/.parquet
      |  |
      |  |-recommendations.csv/.parquet
      |
      |-figures:
      |
      |-results:

    Dataset-Name-B:
      |
      |-dataset:
      |  |
      |  |-item_metadata.csv/.parquet
      |  |
      |  |-recommendations.csv/.parquet
      |
      |-figures:
      |
      |-results:
          
'''

def main():
    
    az.style.use("arviz-darkgrid")
    
    print('possible directories:')
    
    directories = os.scandir(os.getcwd())
    for entry in directories:
        if entry.is_dir():
            print(entry.name)
    
    print('')
    print('dataset-name (name of directory)')
    base_folder = str(input())
    
    print('run Add-Topics-to-Articles to create item_metadata_w_tags-file? (y | n)')
    input1 = input()
    if str(input1) == "y": 
        atta = True
    elif str(input1) == "n":
        atta = False
    else:
        print('invalid input - restart program')
        trlts = False
    
    print('run Transform-Recommendation-Log-to-Samples to create samples_for_model-file? (y | n)')
    input2 = input()
    timestampunit = ''
    if str(input2) == "y": 
        trlts = True
        print('inpurt recommendation-timestamp unit (s | ms)')
        timestampunit = str(input())
    elif str(input2) == "n":
        trlts = False
    else:
        print('invalid input - restart program')
        trlts = False
    
    starttime = datetime.now()
    
    # Add-Topics-to-Articles using BERTopic
    
    if atta:
    
        # if data in csv-file -> provide parquet-file
        if os.path.exists(f"{base_folder}/dataset/item_metadata.csv"):  
            temp_df = pd.read_csv(f"{base_folder}/dataset/item_metadata.csv")
            temp_df.to_parquet(f"{base_folder}/dataset/item_metadata.parquet", index=False)
    
        recommended_articles = pd.read_parquet(
            f"{base_folder}/dataset/item_metadata.parquet")[['item', 'text']]
        
        # Change to the correct language
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=10,
            min_samples=10,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        topic_model = BERTopic(
            language='english',  # Change this to multilingual when text is non-english
            min_topic_size=10,
            vectorizer_model=CountVectorizer(stop_words='english', ngram_range=(1, 2)),
            hdbscan_model=hdbscan_model
        )
        
        docs = recommended_articles["text"].values
        topics, probs = topic_model.fit_transform(docs)
        recommended_articles["tag"] = topics
        
        # Store the model so we can use it in the future.
        model_name = "bertopic_base_model"
        topic_model.save(f"{base_folder}/{model_name}")
        
        recommended_articles[["item", "text", "tag"]].to_parquet(
            f"{base_folder}/item_metadata_w_tags.parquet", index=False)
    
        print('Add-Topics-to-Articles - finished', datetime.now())
        
    else:
        print('Add-Topics-to-Articles - skipped', datetime.now())
    
    # Transform-Recommendation-Log-to-Samples
    
    if trlts:
        
        
        # if data in csv-file -> provide parquet-file
        if os.path.exists(f"{base_folder}/dataset/recommendations.csv"):  
            temp_df = pd.read_csv(f"{base_folder}/dataset/recommendations.csv")
            temp_df.to_parquet(f"{base_folder}/dataset/recommendations.parquet", index=False)
        
        recommendations_df = pd.read_parquet(f"{base_folder}/dataset/recommendations.parquet")
        item_metadata_df = pd.read_parquet(f"{base_folder}/item_metadata_w_tags.parquet")
        
        # Get datetime from epoch
        recommendations_df['datetime'] = pd.to_datetime(
            recommendations_df["timestamp"], unit=timestampunit)
        
        # Calculate user information
        user_df = pd.to_datetime(recommendations_df.groupby(
            'user').datetime.min().rename('signup_date').dt.date).reset_index()
        
        
        user_df.to_parquet(f'{base_folder}/user_information.parquet', index=True)
        
        # Add user info to recommendations
        augmented_reco_df = pd.merge(
            recommendations_df, user_df, how="left", on="user", validate="many_to_one")
        
        # Add date
        augmented_reco_df["date"] = pd.to_datetime(
            augmented_reco_df["datetime"].dt.date)

        # Assign index to unique days in the dataset
        min_date = augmented_reco_df["date"].min()
        max_date = augmented_reco_df["date"].max()
        
        min_day = (min_date.isocalendar().week - 1) * 7 + min_date.isocalendar().weekday
        min_year = min_date.isocalendar().year
        max_day = (max_date.isocalendar().week - 1) * 7 + max_date.isocalendar().weekday
        max_year = max_date.isocalendar().year
        n_days = (max_year - min_year) * 365 + (max_day - min_day) + 1
        min_year, min_day, max_year, max_day, n_days
        
        day_index_map = {(min_day - 1 + i) % 365 + 1: i for i in range(n_days)}
        # min_day, max_day, day_index_map
        
        
        def assign_day_index(x):
            try:
                return day_index_map[x]
            except:
                print('EXCEPT:', x)  
        
        
        augmented_reco_df["day_index"] = ((augmented_reco_df["date"].dt.isocalendar(
        ).week - 1) * 7 + augmented_reco_df["date"].dt.isocalendar().day).map(assign_day_index)
        
        # Add days since user signed up
        augmented_reco_df["days_since_signup"] = (
            (augmented_reco_df["date"] - augmented_reco_df["signup_date"]).dt.days).astype(int)
        
        # Plot how many recommendations were made in each day
        augmented_reco_df["date"].dt.isocalendar().day.hist()
        plt.savefig(f'{base_folder}/figures/recs_each_day.png', dpi='figure')
        plt.clf()
        
        # How many recommendations made in each day since signup
        augmented_reco_df["days_since_signup"].plot.hist()
        plt.savefig(f'{base_folder}/figures/recs_each_day_since_signup.png', dpi='figure')
        plt.clf()
        
        # Frequency of every day index in the dataset
        augmented_reco_df["day_index"].plot.hist()
        plt.savefig(f'{base_folder}/figures/freq_of_dayIndex.png', dpi='figure')
        plt.clf()
        
        # Write intermediate data to folder for reuse
        augmented_reco_df.to_parquet(f"{base_folder}/augmented_reco_df.parquet", index=False)
        
        augmented_reco_df = pd.merge(augmented_reco_df, item_metadata_df, on="item")
        
        augmented_reco_df
        
        recommendations_grouped_day = augmented_reco_df.groupby(
            ["user", "day_index", "days_since_signup"]).agg({"item": lambda x: len(set(x)), "tag": lambda y: len(set(y))}).reset_index().rename(columns={"item": "count", "tag": "variety"})
        
        recommendations_grouped_day.to_parquet(
            f"{base_folder}/samples_for_model.parquet",
            index=False,
        )
    
        print('Transform-Recommendation-Log-to-Samples - finished', datetime.now())
    
    else:
        print('Transform-Recommendation-Log-to-Samples - skipped', datetime.now())
    
    # Fit-Model
    
    df = pd.read_parquet(f"{base_folder}/samples_for_model.parquet")
    
    # GLMM model configuration
    # user-specific random slope omitted (to add it back, add "+ (1|user)")
    formula = """
    variety ~ 
    1 
    + (days_since_signup) 
    + (np.log(count))
    + (1|day_index)
    """
    
    # Parameters for the experiment
    draws = 1000
    tune = 3000
    
    #
    model = bmb.Model(
        formula = formula,
        data = df,
        family="negativebinomial",
        link="log",
        dropna=True,
        auto_scale=True
    )
    
    trace = model.fit(draws=draws, tune=tune, target_accept=0.9)
    
    model.plot_priors()
    plt.savefig(f'{base_folder}/figures/plot_priors.pdf', dpi=300)
    
    az.plot_trace(trace, var_names=["Intercept", "days_since_signup",
                  "np.log(count)", "1|day_index", "1|day_index_sigma"])
    plt.savefig(f'{base_folder}/figures/plot_trace.pdf', dpi=300)
    
    exp_coeff_table = az.summary(
        np.exp(trace.posterior), var_names=[
            "Intercept",
            "days_since_signup",
        ]
    )
    
    exp_coeff_table
    
    exp_coeff_table.to_parquet(
        f"{base_folder}/results/exp_coeff_table.parquet", index=True)
    
    coeff_table = az.summary(
        trace.posterior, var_names=[
            "Intercept",
            "days_since_signup",
            "np.log(count)",
            "1|day_index_sigma",
            "variety_alpha"
        ]
    )
    
    coeff_table
    
    coeff_table.to_parquet(f"{base_folder}/results/other_coeff_table.parquet", index=True)
    
    endtime = datetime.now()
    deltatime = endtime - starttime
    timedf = pd.DataFrame({'starttime': [starttime],
                           'endtime': [endtime],
                           'deltatime': [deltatime]})
    timedf.to_parquet(f"{base_folder}/time_stats.parquet", index=False)
    
    print('Fit-Model - finished', datetime.now())
    
if __name__ == '__main__':
    main()