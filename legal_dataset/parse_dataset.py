
import json


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
            conclusions = [c for c in argument['conclusion']] if type (argument['conclusion']) == list else [argument['conclusion']]
            argument_premises = [clauses_id_to_text[p] for p in premises]
            argument_conclusions = [clauses_id_to_text[c] for c in conclusions]
            arguments.append({'premises': argument_premises, 'conclusion': argument_conclusions})

        all_data_arguments.append(arguments)
    print(1)


if __name__ == '__main__':
    read_data()