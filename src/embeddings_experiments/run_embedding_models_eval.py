import os
import torch
import sys
import time
import json
from statistics import mean

from dotenv import load_dotenv
from src.datasets.local_dataset import load_datasets_base, URL_DATASETS, get_dataframe_content
from src.embedding_models.embedding_models import EMBEDDING_MODELS, LocalHuggingFaceEmbeddingFunction
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from chromadb import PersistentClient
from tqdm import tqdm
from src.rag_experiments.rag_utils import compute_total_tokens




if __name__ == '__main__':

    load_dotenv()

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    collections_dir = '../../data/collections/'
    results_dir = '../../data/results/embeddings_eval/'

    os.makedirs(collections_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f'\nTotal embedding models: {len(EMBEDDING_MODELS)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {device}')

    print('\nLoading datasets')

    list_datasets = load_datasets_base(URL_DATASETS)

    print(f'\nTotal datasets: {len(list_datasets)}')

    list_topn = [1, 3, 5, 10, 20]
    # list_topn = [5]

    for cont_ds, dataset in enumerate(list_datasets, start=1):

        dataframe = dataset['df']

        # if dataset['url'] in ['pira', 'FairytaleQA-translated-ptBR']:
        #     continue

        print(f'\nDataset {cont_ds}/{len(list_datasets)}: {dataset["url"]} -- {len(dataframe)}')

        collections_dataset_dir = os.path.join(collections_dir, dataset['url'])

        os.makedirs(collections_dataset_dir, exist_ok=True)

        ids, documents = get_dataframe_content(dataframe)

        for cont_emb, embedding_model in enumerate(EMBEDDING_MODELS, start=1):

            print(f'\n\tEmbedding Model {cont_emb}/{len(EMBEDDING_MODELS)}: {embedding_model["name"]}')

            results_dataset_dir = os.path.join(results_dir, dataset['url'])

            os.makedirs(results_dataset_dir, exist_ok=True)

            results_file_path = os.path.join(results_dataset_dir, f'{embedding_model["name"]}.json')

            if os.path.exists(results_file_path):
                continue

            collection_name = f'{dataset["url"]}_{embedding_model["name"]}'

            if embedding_model['ef'] == 'LocalHuggingFaceEmbeddingFunction':
                embedding_function = LocalHuggingFaceEmbeddingFunction(embedding_model['model'])
            else:
                embedding_function = OpenAIEmbeddingFunction(
                    api_key=OPENAI_API_KEY,
                    model_name=embedding_model['model'])

            client = PersistentClient(path=collections_dataset_dir)

            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={'hnsw:space': 'cosine'},
                embedding_function=embedding_function
            )

            collection.upsert(
                ids=ids,
                documents=documents
            )

            dict_results_model = {}

            for top_n in list_topn:

                print(f'\n\t\tTop-{top_n}\n')

                hits = 0
                list_token_size = []

                list_precisions = []
                list_recalls = []

                list_results = []

                initial_time = time.time()

                with tqdm(total=len(dataframe), file=sys.stdout, colour='blue',
                          desc='\t\t  Recall: 0.0') as pbar:

                    cont = 1

                    for _, row in dataframe.iterrows():

                        results = collection.query(
                            query_texts=[row['question']],
                            n_results=top_n
                        )

                        true_context = [row['context']]

                        retrieved_contexts = results['documents'][0]

                        is_retrieved_context = False

                        if row['context'] in set(retrieved_contexts):
                            is_retrieved_context = True
                            hits += 1

                        list_results.append(
                            {
                                'is_retrieved_context': is_retrieved_context,
                                'question': row['question'],
                                'ground_truth_context': row['context'],
                                'retrieved_context': retrieved_contexts
                            }
                        )

                        total_tokens = compute_total_tokens(retrieved_contexts)

                        list_token_size.append(total_tokens)

                        recall = hits / cont

                        cont += 1

                        tokens_size_sum =  sum(list_token_size)
                        tokens_size_mean = mean(list_token_size)

                        pbar.set_description(f'\t\t  Recall: {recall:.3f} - '
                                             f'Total Token: {tokens_size_sum:_.2f} - '
                                             f'Média Token: {tokens_size_mean:_.2f}')

                        pbar.update(1)

                final_time = time.time()

                execution_time = final_time - initial_time

                recall = hits / len(dataframe)

                dict_results_model[top_n] = {
                    'total_examples': len(dataframe),
                    'hits': hits,
                    'total_tokens_retrieved_contexts': tokens_size_sum,
                    'mean_tokens_retrieved_contexts': tokens_size_mean,
                    'recall': recall,
                    'time': execution_time,
                    'results': list_results
                }

            with open(file=results_file_path, mode='w', encoding='utf-8') as json_file:
                json.dump(dict_results_model, json_file, indent=4)
