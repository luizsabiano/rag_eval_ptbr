import os
import json

"""
    TO DO:
    
        Ver algo assim: https://medium.com/@marketing_novita.ai/how-to-get-token-count-in-python-for-cost-optimization-e681fb844586
        
        Incluir uma coluna com o total de tokens por top_n.
        
"""

if __name__ == '__main__':

    results_dir = '../../data/results/embeddings_eval/'

    list_dataset_names = os.listdir(results_dir)

    dict_results = {}

    for dataset_name in list_dataset_names:

        dataset_results_dir = os.path.join(results_dir, dataset_name)

        if os.path.isfile(dataset_results_dir):
            continue

        print(f'\nDataset: {dataset_name}')

        list_model_names = os.listdir(dataset_results_dir)

        for model_name in list_model_names:

            if model_name not in [
                'multilingual_e5_small.json',
                'multilingual_e5_base.json',
                'multilingual_e5_large.json',
                'OpenAI_3_small.json',
                'OpenAI_3_large.json'
            ]:
                continue

            print(f'\n\tModel Name: {model_name}')

            file_path = os.path.join(dataset_results_dir, model_name)

            with open(file=file_path, mode='r', encoding='utf-8') as json_file:
                results_dict = json.load(json_file)

            dict_top_n = {}

            for top_n in results_dict.keys():

                if top_n == '3':
                    continue

                dict_top_n[top_n] = (
                    results_dict[top_n]['recall'],
                    results_dict[top_n]['total_tokens_retrieved_contexts'],
                    results_dict[top_n]['mean_tokens_retrieved_contexts']
                )

            if dataset_name not in dict_results:
                dict_results[dataset_name] = {}

            model_name = model_name.replace('.json', '').strip()

            dict_results[dataset_name][model_name] = dict_top_n

    csv_content = 'Dataset;Model;Topn;Recall;Total_Token;Mean_Token\n'

    for dataset_name, dict_model_results in dict_results.items():

        print(f'\nDataset: {dataset_name}')

        for model_name, model_results in dict_model_results.items():

            print(f'\n\tModel: {model_name}\n')

            for top_n, statistics in model_results.items():

                print(f'\t\tTop-{top_n} -- Recall: {statistics[0]:.3f} -- Total Token : {statistics[1]: _.0f} -- Media Token : {statistics[2]: _.0f} ')

                csv_content += f'{dataset_name};{model_name};{top_n};{statistics[0]};{statistics[1]};{statistics[2]}\n'

    csv_content = csv_content[:-1]

    csv_results_file_path = os.path.join(results_dir, 'results.csv')

    with open(file=csv_results_file_path, mode='w') as file:
        file.write(csv_content)
