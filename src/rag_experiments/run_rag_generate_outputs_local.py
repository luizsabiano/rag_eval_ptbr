import os

from dotenv import load_dotenv
from src.datasets.local_dataset import load_datasets_base, URL_DATASETS
from langchain_openai import ChatOpenAI
from src.rag_experiments.rag_utils import get_huggingface_llm, run_experiment_dataset


if __name__ == '__main__':

    load_dotenv()

    collections_dir = '../../data/collections/'
    rag_outputs_dir = '../../data/rag/outputs/'

    prompt_file_path = '../../data/resources/prompt.txt'

    os.makedirs(rag_outputs_dir, exist_ok=True)

    SAMBA_NOVA_API_KEY = os.getenv('SAMBA_NOVA_API_KEY_4')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    SABIA_API_KEY = os.getenv('SABIA_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    HF_TOKEN = os.getenv('HF_TOKEN')

    max_tokens_answer = 128

    topn_chroma = 5

    URL_DATASETS = [
        'paulopirozelli/pira',
        'benjleite/FairytaleQA-translated-ptBR',
        'tiagofvb/squad2-pt-br-no-impossible-questions',
    ]

    # Descomente os modelos que deseja avaliar.
    list_llm_models = [
        # {
        #     'name': 'llama_32_1b',
        #     'checkpoint': 'meta-llama/Llama-3.2-1B-Instruct',
        #     'cloud': 'local',
        #     'base_url': None
        # },
        # {
        #     'name': 'llama_32_3b',
        #     'checkpoint': 'meta-llama/Llama-3.2-3B-Instruct',
        #     'cloud': 'local',
        #     'base_url': None
        # },
        # {
        #     'name': 'llama_32_8b',
        #     'checkpoint': 'meta-llama/Meta-Llama-3-8B-Instruct',
        #     'cloud': 'local',
        #     'base_url': None
        # },
        # {
        #     'name': 'gemma_2_2b',
        #     'checkpoint': 'google/gemma-2-2b-it',
        #     'cloud': 'local',
        #     'base_url': None
        # },
        {
            'name': 'gemma_2_9b',
            'checkpoint': 'google/gemma-2-9b-it',
            'cloud': 'local',
            'base_url': None
        },
    ]

    list_embedding_models = [
        {
            'name': 'multilingual_e5_large',
            'ef': 'LocalHuggingFaceEmbeddingFunction',
            'model': 'intfloat/multilingual-e5-large'
        }
    ]

    with open(file=prompt_file_path, mode='r') as prompt_file:
        PROMPT_TEMPLATE = prompt_file.read()

    list_datasets = load_datasets_base(URL_DATASETS)

    print('\nRunning Experiment')

    time_sleep = 1

    for llm_model_dict in list_llm_models:

        print(f'\n\tLLM: {llm_model_dict["name"]}')

        llm_model_dict['model'] = get_huggingface_llm(
            llm_checkpoint=llm_model_dict['checkpoint'],
            max_new_tokens=max_tokens_answer,
            api_key=HF_TOKEN
        )

        for embedding_model_dict in list_embedding_models:

            print(f'\n\t\tEmbedding: {embedding_model_dict["name"]}')

            for dataset in list_datasets:

                print(f'\n\t\t\tDataset: {dataset["url"]}\n')

                rag_dataset_outputs_dir = os.path.join(rag_outputs_dir, dataset["url"])

                os.makedirs(rag_dataset_outputs_dir, exist_ok=True)

                try:

                    run_experiment_dataset(dataset, embedding_model_dict, llm_model_dict, OPENAI_API_KEY,
                                           topn_chroma, PROMPT_TEMPLATE, time_sleep, collections_dir,
                                           rag_dataset_outputs_dir)
                except Exception as e:
                    print(f'\n\t\t\t\tDataset: {e}\n')
