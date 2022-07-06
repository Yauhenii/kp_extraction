import pandas as pd
import os
import ast

import torch



from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

min_words_count = 5
max_words_count = 20
run_sbert = False

SIMILARITY_THRESHOLD = 0.35
CONTRADICT = -1
SUPPORT = 1


def get_all_sentences_df(test_args_with_q_scores):
    sentences, scores, topics = [], [], []
    for index, row in test_args_with_q_scores.iterrows():
        sents = ast.literal_eval(row["sents_with_scores"])
        for sent, score in sents:
            sentences.append(sent)
            scores.append(score)
            topics.append(row['topic'])

    all_sentences_df = pd.DataFrame({"topic": topics, "sentence": sentences, "q_score": scores})
    all_sentences_df['word_count'] = all_sentences_df['sentence'].str.split().str.len()
    return all_sentences_df


def get_high_quality_sentences(all_sentences_df):
    return all_sentences_df[all_sentences_df['q_score'] > 0.5]


def get_sentences_in_range_of_num_words(all_sentences_df):
    mask = (all_sentences_df['word_count'] > min_words_count) & (all_sentences_df['word_count'] < max_words_count)
    all_sentences_df = all_sentences_df.loc[mask]
    return all_sentences_df

def get_cross_sentences(all_sentences_df):
    unique_topics = all_sentences_df['topic'].unique()

    arg1, arg2 = [], []
    q_score_1, q_score_2 = [], []
    word_count1, word_count2 = [], []
    topics = []

    for t in unique_topics:
        df_f = all_sentences_df[all_sentences_df['topic'] == t]
        unique_sents = df_f["sentence"].unique()
        for s1 in unique_sents:
            row_1 = df_f[df_f['sentence'] == s1].iloc[0]
            for s2 in unique_sents:
                row_2 = df_f[df_f['sentence'] == s2].iloc[0]
                arg1.append(s1)
                arg2.append(s2)
                topics.append(row_1['topic'])
                q_score_1.append(row_1['q_score'])
                q_score_2.append(row_2['q_score'])
                word_count1.append(row_1['word_count'])
                word_count2.append(row_2['word_count'])

    cross_df = pd.DataFrame({
        "sent1": arg1,
        "sent2": arg2,
        "topic": topics,
        "q_score_1": q_score_1,
        "q_score_2": q_score_2,
        "word_count1": word_count1,
        "word_count2": word_count2
    })
    return cross_df





def cos_sim(a, b):
    """ Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j. :return: Matrix with res[i][j]  = cos_sim(a[i], b[j]) """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return res



def get_pair_sentences_score_df(cross_sentences_df):
    # Two lists of sentences
    sentences1 = cross_sentences_df['sent1']
    sentences2 = cross_sentences_df['sent2']

    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True, batch_size=8, show_progress_bar=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True, batch_size=8, show_progress_bar=True)

    cosine_scores = cos_sim(embeddings1.cpu(), embeddings2.cpu())

    sent1 = []
    sent2 = []
    scores = []

    # Output the pairs with their score
    for i in range(len(sentences1)):
        sent1.append(sentences1[i])
        sent2.append(sentences2[i])
        scores.append(cosine_scores[i][i].item())

    res_df = pd.DataFrame({"sent1": sent1, "sent2": sent2, "score": scores})
    res_df = res_df[res_df['score'] > SIMILARITY_THRESHOLD]
    res_df = res_df[res_df['sent1'] != res_df['sent2']]

    return res_df

def calc_cosine_sim(argument, topic):
    argument_emb = model.encode(argument, convert_to_tensor=True)
    topic_emb = model.encode(topic, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(argument_emb, topic_emb)
    cosine_score = round(cosine_score.item(), 3)

    return cosine_score

def main():
    os.chdir('../../legal_dataset')
    if run_sbert:
        test_args_with_q_scores = pd.read_csv(r'args_topic_score.csv', index_col=None)
        arguments, scores, topics = [], [], []
        for argument, topic, score in zip(test_args_with_q_scores['sentence_text'], test_args_with_q_scores['kp'], test_args_with_q_scores['match_score']):
            if topic == "none":
                continue
            scores.append(score)
            arguments.append(argument)
            topics.append(topic)

        all_sentences_df = pd.DataFrame({"topic": topics, "sentence": arguments, "q_score": scores})
        all_sentences_df['word_count'] = all_sentences_df['sentence'].str.split().str.len()
        all_sentences_df.to_csv(r'args_topic_emb_score.csv', index=False)

        all_sentences_df = pd.read_csv(r'args_topic_emb_score.csv', index_col=None)
        all_sentences_df = get_high_quality_sentences(all_sentences_df)
        all_sentences_df = get_sentences_in_range_of_num_words(all_sentences_df)
        all_sentences_df = all_sentences_df.drop_duplicates(["topic", "sentence"])

        cross_sentences_df = get_cross_sentences(all_sentences_df)
        pair_sentences_df = get_pair_sentences_score_df(cross_sentences_df)

        res_df_enhanced = pd.merge(pair_sentences_df, all_sentences_df, left_on="sent1", right_on="sentence", how="inner")
        res_df_enhanced = res_df_enhanced.rename(columns={'q_score': 'q_score_1', 'word_count': 'word_count_1'})
        res_df_enhanced = pd.merge(res_df_enhanced, all_sentences_df, left_on="sent2", right_on="sentence", how="inner")
        res_df_enhanced = res_df_enhanced.rename(columns={'q_score': 'q_score_2', 'word_count': 'word_count_2'})
        res_df_enhanced = res_df_enhanced[
        ['sent1', 'sent2', 'score', 'topic_x', 'q_score_1', 'word_count_1', 'q_score_2', 'word_count_2']]
        res_df_enhanced = res_df_enhanced.rename(columns={'topic_x': 'topic', 'stance_x': 'stance'})
        res_df_enhanced.to_csv(r'pot_kps_cross_with_sim_scores_df.csv', index=False)

    res_df_enhanced = pd.read_csv(r'pot_kps_cross_with_sim_scores_df.csv', index_col=None)
    print(1)
    # get count grouped by topic, stance , if tie break with q score
    match_count_df = res_df_enhanced[res_df_enhanced['score'] > SIMILARITY_THRESHOLD]
    match_count_df = match_count_df[match_count_df['sent1'] != match_count_df['sent2']]
    only_counts_df = match_count_df.groupby(by='sent1').size().reset_index(name='counts')
    match_count_df = pd.merge(match_count_df, only_counts_df, on='sent1')
    match_count_df = match_count_df[
        ['sent1', 'topic', 'q_score_1', 'word_count_1', 'counts']].drop_duplicates()
    N = 20
    top_df = match_count_df.groupby(by=["topic"]).apply(
        lambda df: df.sample(frac=1, random_state=42).sort_values(by=["counts", "q_score_1", "word_count_1"],
                                                                  ascending=(False, False, True))).reset_index(
        drop=True)
    print(1)
    # filter redundant sentences

    FILTER_THRESHOLD = 0.4

    top_filtered_sents = set()

    for index, row in top_df.iterrows():
        s1 = row['sent1']
        if len(top_filtered_sents) == 0:
            top_filtered_sents.add(s1)
            continue

        sim_sentences = res_df_enhanced[(res_df_enhanced.sent1 == s1)]
        sim_sentences = sim_sentences[(sim_sentences.sent2.isin(top_filtered_sents))]
        if len(sim_sentences) == 0:
            top_filtered_sents.add(s1)
            continue
        if max(sim_sentences['score']) < FILTER_THRESHOLD:
            top_filtered_sents.add(s1)

    top_filtered_df = pd.DataFrame({"sent1": list(top_filtered_sents)})
    top_filtered_df = pd.merge(top_filtered_df, top_df, on='sent1')
    N = 20
    top_filtered_df = top_filtered_df.groupby(by=["topic"]).apply(
        lambda df: df.sample(frac=1, random_state=42).sort_values(by=["counts", "q_score_1", "word_count_1"],
                                                                  ascending=(False, False, True)).head(N)).reset_index(
        drop=True)
    print(1)
if __name__ == '__main__':
    main()