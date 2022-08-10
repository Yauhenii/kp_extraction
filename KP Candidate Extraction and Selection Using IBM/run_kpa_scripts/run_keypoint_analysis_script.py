
import logging
from debater_python_api.api.clients.keypoints_client import KpAnalysisTaskFuture, \
    KpaIllegalInputException, KpAnalysisUtils, KpAnalysisClient
from debater_python_api.api.debater_api import DebaterApi
import pandas as pd

DEBATER_API_KEY = 'fc37b734a3d45adea830254ab316551eL05'
HOST = 'https://keypoint-matching-backend.debater.res.ibm.com'
DOMAIN = 'legal_dataset'

# threshold 0.9
# no KPS texts: 4, 5, 6, 9, 10, 12, 13, 16, 17, 18, 19, 23
input_csv_per_text = [  '../dataset/text_00_echr_arguments.csv', '../dataset/text_01_echr_arguments.csv',
                '../dataset/text_02_echr_arguments.csv', '../dataset/text_03_echr_arguments.csv',
                '../dataset/text_07_echr_arguments.csv', '../dataset/text_08_echr_arguments.csv',
                '../dataset/text_11_echr_arguments.csv', '../dataset/text_14_echr_arguments.csv',
                '../dataset/text_15_echr_arguments.csv', '../dataset/text_20_echr_arguments.csv',
                '../dataset/text_21_echr_arguments.csv', '../dataset/text_22_echr_arguments.csv',
                '../dataset/text_24_echr_arguments.csv', '../dataset/text_25_echr_arguments.csv',
                '../dataset/text_26_echr_arguments.csv', '../dataset/text_27_echr_arguments.csv',
                '../dataset/text_28_echr_arguments.csv', '../dataset/text_29_echr_arguments.csv',
                '../dataset/text_30_echr_arguments.csv', '../dataset/text_31_echr_arguments.csv',
                '../dataset/text_32_echr_arguments.csv', '../dataset/text_33_echr_arguments.csv',
                '../dataset/text_34_echr_arguments.csv', '../dataset/text_35_echr_arguments.csv',
                '../dataset/text_37_echr_arguments.csv', '../dataset/text_38_echr_arguments.csv',
                '../dataset/text_39_echr_arguments.csv', '../dataset/text_40_echr_arguments.csv',
                '../dataset/text_41_echr_arguments.csv', '../dataset/text_42_echr_arguments.csv']

input_csv_all_text = ['../dataset/echr_arguments.csv']

output_csv_per_text = ['../dataset/text_00_echr_arguments_kpa_results.csv',
                '../dataset/text_01_echr_arguments_kpa_results.csv',
                '../dataset/text_02_echr_arguments_kpa_results.csv', '../dataset/text_03_echr_arguments_kpa_results.csv',
                '../dataset/text_07_echr_arguments_kpa_results.csv', '../dataset/text_08_echr_arguments_kpa_results.csv',
                '../dataset/text_11_echr_arguments_kpa_results.csv', '../dataset/text_14_echr_arguments_kpa_results.csv',
                '../dataset/text_15_echr_arguments_kpa_results.csv', '../dataset/text_20_echr_arguments_kpa_results.csv',
                '../dataset/text_21_echr_arguments_kpa_results.csv', '../dataset/text_22_echr_arguments_kpa_results.csv',
                '../dataset/text_24_echr_arguments_kpa_results.csv', '../dataset/text_25_echr_arguments_kpa_results.csv',
                '../dataset/text_26_echr_arguments_kpa_results.csv', '../dataset/text_27_echr_arguments_kpa_results.csv',
                '../dataset/text_28_echr_arguments_kpa_results.csv', '../dataset/text_29_echr_arguments_kpa_results.csv',
                '../dataset/text_30_echr_arguments_kpa_results.csv', '../dataset/text_31_echr_arguments_kpa_results.csv',
                '../dataset/text_32_echr_arguments_kpa_results.csv', '../dataset/text_33_echr_arguments_kpa_results.csv',
                '../dataset/text_34_echr_arguments_kpa_results.csv', '../dataset/text_35_echr_arguments_kpa_results.csv',
                '../dataset/text_37_echr_arguments_kpa_results.csv', '../dataset/text_38_echr_arguments_kpa_results.csv',
                '../dataset/text_39_echr_arguments_kpa_results.csv', '../dataset/text_40_echr_arguments_kpa_results.csv',
                '../dataset/text_41_echr_arguments_kpa_results.csv', '../dataset/text_42_echr_arguments_kpa_results.csv']


output_csv_all_text = ['../dataset/echr_arguments_kpa_results.csv']

output_file_per_text_concat = '../dataset/results_per_text_concat.csv'
output_file_all_text = '../dataset/results_all_text.csv'


def write_sentences_to_csv(sentences, out_file):
    if len(sentences) == 0:
        logging.info('there are no sentences, not saving file')
        return

    cols = list(sentences[0].keys())
    rows = [[s[col] for col in cols] for s in sentences]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_file, index=False)


def get_comments_ids_and_texts(file, ids_column, comment_text_column):
    df = pd.read_csv(file)
    id_text = sorted(list(zip(df[ids_column], df[comment_text_column])))
    id_text = [t for t in id_text if len(t[1]) < 1000]  # comments must have at most 1000 chars
    comments_ids = [str(t[0]) for t in id_text]
    comments_texts = [str(t[1]) for t in id_text]
    return comments_ids, comments_texts


def run_debater(input_csv_files, comment_ids_column, comment_text_column, mapping_threshold):
    KpAnalysisUtils.init_logger()

    # ======================= update params =======================
    api_key = DEBATER_API_KEY
    files = input_csv_files
    domain = DOMAIN
    host = HOST
    # ======================= update params =======================

    debater_api = DebaterApi(apikey=api_key)
    keypoints_client = KpAnalysisClient(apikey=api_key, host=host)
    # keypoints_client.delete_all_domains_cannot_be_undone()

    # what to run
    delete_all_domains = False  # deletes-all! cannot be undone
    restart = True
    delete_domain = True
    create_domain = True
    upload_comments = restart
    run_kpa = True
    download_sentences = False
    get_report = False
    connect_to_running_job = False
    cancel_all_running_jobs_in_the_domain = False

    # how to run
    domain_params = {}
    run_params = {'sentence_to_multiple_kps': False, 'mapping_threshold': mapping_threshold, }
    for file in files:
        if delete_all_domains:
            answer = input("Are you sure you want to delete ALL domains (cannot be undone)? yes/no:")
            if answer == 'yes':
                keypoints_client.delete_all_domains_cannot_be_undone()

        if get_report:
            KpAnalysisUtils.print_report(keypoints_client.get_full_report())

        if delete_domain:
            keypoints_client.delete_domain_cannot_be_undone(domain)
            # answer = input(f'Are you sure you want to delete domain: {domain} (cannot be undone)? yes/no:')
            # if answer == 'yes':
            #     keypoints_client.delete_domain_cannot_be_undone(domain)

        if create_domain:
            try:
                keypoints_client.create_domain(domain, {})
                print('domain was created')
            except KpaIllegalInputException as e:
                print('domain already exist')


        if upload_comments:
            comments_ids, comments_texts = get_comments_ids_and_texts(file, comment_ids_column, comment_text_column)

            keypoints_client.upload_comments(domain=domain,
                                             comments_ids=comments_ids,
                                             comments_texts=comments_texts)

            keypoints_client.wait_till_all_comments_are_processed(domain=domain)

        if download_sentences:
            sentences = keypoints_client.get_sentences_for_domain(domain)
            write_sentences_to_csv(sentences, file.replace('.csv', '_sentences.csv'))

        future = None
        if run_kpa:
            keypoints_client.wait_till_all_comments_are_processed(domain=domain)
            future = keypoints_client.start_kp_analysis_job(domain=domain, run_params=run_params)

        if connect_to_running_job:
            future = KpAnalysisTaskFuture(keypoints_client, '<job_id>')

        if future is not None:
            kpa_result = future.get_result(high_verbosity=True, polling_timout_secs=30)
            for kp_match_d in kpa_result['keypoint_matchings']:
                for match_dict in kp_match_d['matching']:
                    match_dict['comment_id'] = file[16:18] + "#" + match_dict['comment_id']

            KpAnalysisUtils.write_result_to_csv(kpa_result, file.replace('.csv', '_kpa_results.csv'))

        if cancel_all_running_jobs_in_the_domain:
            keypoints_client.cancel_all_extraction_jobs_for_domain(domain=domain)

def substring_after(s, delim):
    return s.partition(delim)[2]


def run_pipeline_all_texts_separate(run_debater):
    if run_debater:
        run_debater(input_csv_files=input_csv_per_text)
    all_texts_dataframe = pd.DataFrame()
    for filename in output_csv_per_text:
        dataframe = pd.read_csv(filename)
        dataframe = dataframe[dataframe['kp'] != 'none']
        all_texts_dataframe = pd.concat([all_texts_dataframe, dataframe])
        all_texts_dataframe['comment_id'] = all_texts_dataframe['comment_id']

    sorted_df = all_texts_dataframe.sort_values(by=['comment_id', 'sentence_id'])
    concat_sent = sorted_df.groupby(['comment_id'])['sentence_text'].apply(','.join).reset_index()
    merged = pd.merge(sorted_df, concat_sent, on='comment_id')
    merged = merged.sort_values("match_score", ascending=False).groupby(["comment_id"]).first().reset_index()
    res = merged.loc[:, ["kp", "sentence_text_y", "match_score"]]
    res.rename(columns={'kp': 'kp', 'sentence_text_y': 'argument', 'match_score': 'match_score'}, inplace=True)
    res = res.drop_duplicates(keep='first')
    res.to_csv(output_file_per_text_concat, index=False)

def run_pipeline_all_texts_together(run_debater):
    if run_debater:
        run_debater(input_csv_files=input_csv_all_text)
    all_texts_dataframe = pd.DataFrame()

    for filename in output_csv_all_text:
        dataframe = pd.read_csv(filename)
        dataframe = dataframe[dataframe['kp'] != 'none']
        all_texts_dataframe = pd.concat([all_texts_dataframe, dataframe])
        all_texts_dataframe['comment_id'] = all_texts_dataframe['comment_id']
    sorted_df = all_texts_dataframe.sort_values(by=['comment_id', 'sentence_id'])
    concat_sent = sorted_df.groupby(['comment_id'])['sentence_text'].apply(','.join).reset_index()
    merged = pd.merge(sorted_df, concat_sent, on='comment_id')
    merged = merged.sort_values("match_score", ascending=False).groupby(["comment_id"]).first().reset_index()
    res = merged.loc[:, ["kp", "sentence_text_y", "match_score"]]
    res.rename(columns={'kp': 'kp', 'sentence_text_y': 'argument', 'match_score': 'match_score'}, inplace=True)
    res = res.drop_duplicates(keep='first')
    res.to_csv(output_file_all_text, index=False)


if __name__ == '__main__':
    # run_pipeline_all_texts_together(run_debater=True)
    run_pipeline_all_texts_separate(run_debater=False)





