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

    SAMBA_NOVA_API_KEY = os.getenv('SAMBANOVA_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    SABIA_API_KEY = os.getenv('SABIA_API_KEY')
    HF_TOKEN = os.getenv('HF_TOKEN')
    # GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


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
        #     'name': 'sabiazinho_3',
        #     'checkpoint': 'sabiazinho-3',
        #     'cloud': 'maritaca',
        #     'base_url': 'https://chat.maritaca.ai/api'
        # },
        #
        # {
        #     'name': 'sabia_3',
        #     'checkpoint': 'sabia-3',
        #     'cloud': 'maritaca',
        #     'base_url': 'https://chat.maritaca.ai/api',
        # },

        {
            'name': 'llama_33_70b',
            'checkpoint': 'Meta-Llama-3.3-70B-Instruct',
            'cloud': 'samba_nova',
            'base_url': 'https://api.sambanova.ai/v1'
        },

        # {
        #     'name': 'qwen2_72b',
        #     'checkpoint': 'Qwen2.5-72B-Instruct',
        #     'cloud': 'samba_nova',
        #     'base_url': 'https://api.sambanova.ai/v1'
        # },

        # {
        #     'name': 'llama_31_405b',
        #     'checkpoint': 'Meta-Llama-3.1-405B-Instruct',
        #     'cloud': 'samba_nova',
        #     'base_url': 'https://api.sambanova.ai/v1'
        # }

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

    time_sleep = 0

    for llm_model_dict in list_llm_models:

        print(f'\n\tLLM: {llm_model_dict["name"]}')

        if llm_model_dict['cloud'] == 'samba_nova':
            api_key = SAMBA_NOVA_API_KEY
            time_sleep = 10
        elif llm_model_dict['cloud'] == 'maritaca':
            api_key = SABIA_API_KEY

        else:
            print('\n\nError Cloud.')
            exit(-1)

        llm_model_dict['model'] = ChatOpenAI(
            model_name=llm_model_dict['checkpoint'],
            temperature=0.0,
            max_tokens=max_tokens_answer,
            api_key=api_key,
            base_url=llm_model_dict['base_url']
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
