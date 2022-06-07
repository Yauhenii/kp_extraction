import pandas as pd
from summarizer.sbert import SBertSummarizer
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from pplm.run_pplm import run_pplm_example
import ast

nltk.download('stopwords')
model = SBertSummarizer('paraphrase-MiniLM-L6-v2')


def get_extractive_kps(text_input, num_sentences=10):
    if isinstance(text_input, list):
        text = ''
        for t in text_input:
            text += t.strip() + '. '
    else:
        text = text_input
    return model(text, num_sentences=num_sentences)


def get_bow(text):
    # Word Tokenization
    tokenized_word = word_tokenize(text)
    # remove stop word
    stop_words = set(stopwords.words("english"))
    filtered_words = []
    for w in tokenized_word:
        if w not in stop_words:
            filtered_words.append(w)
    # stemming
    ps = PorterStemmer()
    stemmed_words = []
    for w in filtered_words:
        stem = ps.stem(w)
        if len(stem) > 3:
            stemmed_words.append(stem)
    # bow
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
    text_counts = cv.fit_transform(stemmed_words)
    cv_dataframe = pd.DataFrame(text_counts.toarray(), columns=cv.get_feature_names())
    bow = []
    for c in cv_dataframe.columns.to_list():
        if len(c) > 3:
            bow.append(c)

    return bow


def prepare_whitepaper(name, num_kps=4):
    df = pd.read_csv(
        rf'D:\backup_user\crypto\thesis\my-repos\cryptocurrencies-kpa\data\processed\whitepapers\{name}_with_sections.csv',
        index_col=None)
    df["extracted_kps"] = df["text"].apply(lambda x: get_extractive_kps(x, num_kps))
    df.to_csv(f'./kpa_dataset/processed_{name}.csv', index=None)


def prepare_kpa_dataset():
    df = pd.read_csv(
        r'D:\backup_user\crypto\thesis\my-repos\cryptocurrencies-kpa\data\processed\ARG_KP_2021\all_complete.csv')
    df = df[["topic", "stance", "argument", "key_point"]]
    test_df = df.groupby(['topic', 'stance']).agg({
        'argument': lambda x: set([item for item in x]),
        'key_point': lambda x: set([item for item in x])
    }).reset_index()
    test_df["extracted_kps"] = test_df["argument"].apply(lambda x: get_extractive_kps(x))
    test_df.to_csv('./kpa_dataset/processed_kpa.csv', index=None)


def generate_kpa_bows(df):
    # full KPA dataset bow
    full_text = ''
    for i, row in df.iterrows():
        for a in ast.literal_eval(row['argument']):
            full_text += a.strip() + '. '
    full_bow = get_bow(full_text)
    with open('./kpa_dataset/full_bow.txt', 'w') as f:
        for item in full_bow:
            f.write("%s\n" % item)

    # topic bow
    for i, row in df.iterrows():
        full_text = ''
        for a in ast.literal_eval(row['argument']):
            full_text += a.strip() + '. '
        full_bow = get_bow(full_text)
        with open(f"./kpa_dataset/{row['topic']}_{row['stance']}_bow.txt", 'w') as f:
            for item in full_bow:
                f.write("%s\n" % item)


def generate_whitepaper_bows(name, df):
    # full whitepaper bow
    full_text = ''
    for i, row in df.iterrows():
        full_text += row['text'].strip() + '. '
    full_bow = get_bow(full_text)
    with open(f'./kpa_dataset/{name}_full_bow.txt', 'w') as f:
        for item in full_bow:
            f.write("%s\n" % item)

    # section bow
    for i, row in df.iterrows():
        full_bow = get_bow(row['text'].strip())
        with open(f"./kpa_dataset/{name}_{row['section']}_bow.txt", 'w') as f:
            for item in full_bow:
                f.write("%s\n" % item)


def get_pplm_summary(bow_path, promote: str, params: dict):
    pert_gen_texts, unpert_gen_text = run_pplm_example(
        pretrained_model=params['pretrained_model'],
        cond_text=promote,
        uncond=False,
        num_samples=params["num_samples"],
        bag_of_words=bow_path,
        discrim=None,
        discrim_weights=None,
        discrim_meta=None,
        class_label=-1,
        length=params['length'],
        stepsize=params['stepsize'],
        temperature=1.0,
        top_k=10,
        sample=params['sample'],
        num_iterations=params['num_iterations'],
        grad_length=10000,
        horizon_length=1,
        window_length=params['window_length'],
        decay=False,
        gamma=params['gamma'],
        gm_scale=params['gm_scale'],
        kl_scale=params['kl_scale'],
        seed=0,
        no_cuda=False,
        colorama=params['colorama'],
        verbosity='regular'
    )
    print(f"unpert_gen_text: {unpert_gen_text}")
    for i, txt in enumerate(pert_gen_texts):
        print(f"pert_gen_text {i}: {txt}")
    return unpert_gen_text, pert_gen_texts


def generate_summarize_kpa(df, params, is_full_bow=True):
    bow_path = './kpa_dataset/full_bow.txt'

    unpert_gen_list = []
    pert_gen_list = []

    for i, row in df.iterrows():
        if not is_full_bow:
            bow_path = f"./kpa_dataset/{row['topic']}_{row['stance']}_bow.txt"
        unpert_gen_text, pert_gen_texts = get_pplm_summary(bow_path, row['extracted_kps'], params)
        unpert_gen_list.append(
            unpert_gen_text.replace('<|endoftext|>', '').replace("\n", "").replace(row['extracted_kps'], ''))
        prep_texts = [i.replace('<|endoftext|>', '').replace("\n", "").replace(row['extracted_kps'], '') for i in
                      pert_gen_texts]
        pert_gen_list.append(prep_texts)

    df['unpert_gen_text'] = unpert_gen_list
    df['pert_gen_texts'] = pert_gen_list
    df.to_csv(f"./kpa_dataset/generated_kpa_full_{is_full_bow}_bow_{params['pretrained_model']}.csv", index=None)


def generate_summarize_whitepaper(df, params, coin_name, is_full_bow=True):
    bow_path = f'./kpa_dataset/{coin_name}_full_bow.txt'

    unpert_gen_list = []
    pert_gen_list = []

    for i, row in df.iterrows():
        if not is_full_bow:
            bow_path = f"./kpa_dataset/{coin_name}_{row['section']}_bow.txt"
        unpert_gen_text, pert_gen_texts = get_pplm_summary(bow_path, row['extracted_kps'], params)
        unpert_gen_list.append(
            unpert_gen_text.replace('<|endoftext|>', '').replace("\n", "").replace(row['extracted_kps'], ''))
        prep_texts = [i.replace('<|endoftext|>', '').replace("\n", "").replace(row['extracted_kps'], '') for i in
                      pert_gen_texts]
        pert_gen_list.append(prep_texts)

    df['unpert_gen_text'] = unpert_gen_list
    df['pert_gen_texts'] = pert_gen_list
    df.to_csv(f"./kpa_dataset/{coin_name}_generated_full_{is_full_bow}_bow_{params['pretrained_model']}.csv",
              index=None)
