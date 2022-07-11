
import json

import csv
import pandas as pd

output_file_name = 'echr_arguments.csv'

def read_data():
    # Opening JSON file
    f = open('ECHR_Corpus.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    all_data_arguments = []
    for record in data:
        clauses_id_to_text = {}
        text_name = record['name']
        for clause in record['clauses']:
            _id, start, end = clause['_id'], clause['start'], clause['end']
            clause_text = record['text'][start:end]
            clauses_id_to_text[_id] = clause_text

        arguments = []
        for argument in record['arguments']:
            premises = [p for p in argument['premises']] if type(argument['premises']) == list else [argument['premises']]
            # conclusions = [c for c in argument['conclusion']] if type (argument['conclusion']) == list else [argument['conclusion']]
            argument_premises = [clauses_id_to_text[p] for p in premises]
            # argument_conclusions = [clauses_id_to_text[c] for c in conclusions]
            arguments += argument_premises

        for arg in arguments:
            all_data_arguments.append((text_name, arg))
    return all_data_arguments




def write_arguments_to_csv(arguments, output_file_name):
    header = ['id', 'name', 'argument']

    with open(output_file_name, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        id = 1
        for name, argument in arguments:
            data = [id, name, argument]
            writer.writerow(data)
            id += 1
    f.close()





if __name__ == '__main__':
    arguments = read_data()
    write_arguments_to_csv(arguments, output_file_name)
