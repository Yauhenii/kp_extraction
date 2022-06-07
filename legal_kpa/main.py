from debater_python_api.api.debater_api import DebaterApi
from debater_python_api.api.clients.keypoints_client import KpAnalysisUtils

from debater_python_api.api.clients.keypoints_client import KpAnalysisClient
from survey_usecase.austin_utils import init_logger
from survey_usecase.austin_utils import print_results


import os
import csv
import random



DEBATER_API_KEY = 'fc37b734a3d45adea830254ab316551eL05'

def run_example_code():
    debater_api = DebaterApi(DEBATER_API_KEY)
    keypoints_client = debater_api.get_keypoints_client()

    comments_texts = [
        'Cannabis has detrimental effects on cognition and memory, some of which are irreversible.',
        'Cannabis can severely impact memory and productivity in its consumers.',
        'Cannabis harms the memory and learning capabilities of its consumers.',
        'Frequent use can impair cognitive ability.',
        'Cannabis harms memory, which in the long term hurts progress and can hurt people',
        'Frequent marijuana use can seriously affect short-term memory.'
        ,
        'Marijuana is very addictive, and therefore very dangerous'
        'Cannabis is addictive and very dangerous for use.',
        'Cannabis can be very harmful and addictive, especially for young people',
        'Cannabis is dangerous and addictive.'
    ]

    KpAnalysisUtils.init_logger()
    keypoint_matchings = keypoints_client.run(comments_texts)
    KpAnalysisUtils.print_result(keypoint_matchings, print_matches=True)


def run_kpa(sentences, run_params):
    sentences_texts = [sentence['text'] for sentence in sentences]
    sentences_ids = [sentence['id'] for sentence in sentences]

    KpClient = KpAnalysisClient(DEBATER_API_KEY)
    KpClient.upload_comments(domain='austin_demo', comments_ids=sentences_ids, comments_texts=sentences_texts, dont_split=True)
    KpClient.wait_till_all_comments_are_processed(domain='austin_demo')
    future = KpClient.start_kp_analysis_job(domain='austin_demo', comments_ids=sentences_ids, run_params=run_params)
    kpa_result = future.get_result(high_verbosity=True, polling_timout_secs=5)

    return kpa_result, future.get_job_id()


def run_kpa_survey_usecase():

    os.chdir('../survey_usecase')
    file = open('dataset_austin_sentences.csv', 'r')
    lines = file.readlines()
    print('\n'.join(lines[:5]))

    with open('./dataset_austin_sentences.csv') as csv_file:
        reader = csv.DictReader(csv_file)
        sentences = list(reader)

    print('There are %d sentences in the dataset' % len(sentences))
    print('Each sentence is a dictionary with the following keys: %s' % str(sentences[0].keys()))

    sentences_2016 = [sentence for sentence in sentences if sentence['year'] == '2016']
    print('There are %d sentences in the 2016 survey' % len(sentences_2016))
    random.seed(0)
    random_sample_sentences_2016 = random.sample(sentences_2016, 1000)

    init_logger()

    # api_key = DEBATER_API_KEY
    # debater_api = DebaterApi(apikey=api_key)
    # keypoints_client = debater_api.get_keypoints_client()

    kpa_result_random_1000_2016, _ = run_kpa(random_sample_sentences_2016, {'n_top_kps': 20})
    print_results(kpa_result_random_1000_2016, n_sentences_per_kp=2, title='Random sample 2016')



if __name__ == '__main__':
    # run_example_code()
    run_kpa_survey_usecase()

