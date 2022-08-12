# Overview

**dataset:**

This directory contains the datasets used and produced during the method investigation 

**results:** 

This directory contains the results xlsx 

* "echr_arguments_kp_berttopic_peg_l.xls" -- result for the HDBSCAN + Legal Pegasus

* "echr_arguments_kp_berttopic_luhn.xls" -- result for the HDBSCAN + Luhn

* "echr_arguments_kp_berttopic_lsa.xls" -- result for the HDBSCAN + LSA 

* "echr_arguments_kp_berttopic_lex_rank.xls" -- result for the HDBSCAN + Lex Rank

* "echr_arguments_kp_berttopic_kl.xls" -- result for the HDBSCAN + KL-Sum

**other:**

This directory contains the notebook for test/train split needed in the notebook 3_legal_clustering

**notebooks:**

These notebooks conveys the investigation process. Most important notebooks are:
1. 5_legal_clustering_legal_bert.ipynb
2. 6_legal_clustering_topic_bert.ipynb
3. 9_gensim_sum_berttopic.ipynb
4. 10_legal_pegasus_sum_berttopic.ipynb

# Full method's investigation

* 1_KPA_clustering.ipynb


Here we use agglomerative clustering to cluster arguments from [ARG KP dataset](https://github.com/IBM/KPA_2021_shared_task/tree/main/kpm_data). We use ARG KP because each argument has a topic, so we can evaluate how good clusterization works (assuming that each topic forms one cluster). 


* 2_KPA_clustering_fine_tunning.ipynb


Now, we try to fine-tune agglomerative clustering implementation from **1_KPA_clustering.ipynb** and evaluate our results on test data from ARG KP.


* 3_legal_clustering.ipynb


We switch to our ECHR dataset and run agglomerative clustering on it. Note: here input datasets are train.csv and test.csv from /other/O_1_legal_topic_modeling.ipynb


* 4_legal_barh_clustering.ipynb


Here we run agglomerative clustering on BarH results with different thresholds: 0.95 and 0.99.


* 5_legal_clustering_legal_bert.ipynb


Now, we encode arguments using LegalBert to increase clusterization performance. After that, we run agglomerative clustering.


* 6_legal_clustering_topic_bert.ipynb


We switch from agglomerative clustering to HDBSCAN (using [BertTopic](https://github.com/MaartenGr/BERTopic) library).


* 8_pegasus_sum_berttopic.ipynb


Cluster arguments text separately using HDBSCAN. Run Pegasus on resulting clusters.


* 9_gensim_sum_berttopic.ipynb


We try different summarization algorithms like Luhn, KL-SUM, BART, etc. Results are in /results folder


* 10_legal_pegasus_sum_berttopic.ipynb


We run [LegelPegasus](https://huggingface.co/nsi319/legal-pegasus) on clusters from HDBSCAN. Results are in /results folder


* 11_legal_pegasus_rephrase_results.ipynb


Try to rephrase arguments using LegalPegasus to get new arguments
