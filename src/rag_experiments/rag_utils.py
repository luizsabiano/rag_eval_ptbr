import time
import torch
import os
import sys
import json
import spacy

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from langchain_huggingface import HuggingFacePipeline
from src.embedding_models.embedding_models import LocalHuggingFaceEmbeddingFunction
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from chromadb import PersistentClient
from tqdm import tqdm
from langchain.prompts import ChatPromptTemplate


nlp = spacy.load('pt_core_news_sm', disable=['ner'])

def compute_total_tokens(list_texts: list) -> int:
  token_size = 0
  for text in list_texts:
    doc = nlp(text)
    token_size += len([token.orth_ for token in doc])
  return token_size


def get_huggingface_llm(llm_checkpoint: str, max_new_tokens: int, api_key: str) -> HuggingFacePipeline:

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        llm_checkpoint,
        token=api_key
    )

    model_config = AutoConfig.from_pretrained(
        llm_checkpoint,
        trust_remote_code=True,
        max_new_tokens=max_new_tokens
    )

    model = AutoModelForCausalLM.from_pretrained(
        llm_checkpoint,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
    )

    pipe = pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        pad_token_id = tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    return llm


def run_experiment_dataset(dataset: dict, embedding_model_dict: dict, llm_model_dict: dict,
                           openai_api_key: str, top_n: int, prompt_template_base: str, time_sleep: int,
                           collections_dir: str, rag_outputs_dir: str):

    collections_dataset_dir = str(os.path.join(collections_dir, dataset['url']))

    if not os.path.exists(collections_dataset_dir):
        print('Error. Collection does not exist!')
        return

    file_name = f'{llm_model_dict["name"]}_{embedding_model_dict["name"]}_{top_n}.json'

    output_file_path = os.path.join(rag_outputs_dir, file_name)

    time_to_search_collection_model = 0
    time_to_search_llm_model = 0

    ids = []
    questions = []
    answers = []
    correct_contexts = []
    relevant_contexts = []
    llm_answer = []

    if os.path.exists(output_file_path):

        with open(file=output_file_path, mode='r', encoding='utf-8') as json_file:

            outputs = json.load(json_file)

            time_to_search_collection_model = outputs['time_to_search_collection_model']
            time_to_search_llm_model = outputs['time_to_search_llm_model']

            ids = outputs['ids']
            questions = outputs['questions']
            answers = outputs['answers']
            correct_contexts = outputs['correct_contexts']
            relevant_contexts = outputs['relevant_contexts']
            llm_answer = outputs['llm_answer']

    collection_name = f'{dataset["url"]}_{embedding_model_dict["name"]}'

    if embedding_model_dict['ef'] == 'LocalHuggingFaceEmbeddingFunction':
        embedding_function = LocalHuggingFaceEmbeddingFunction(
            embedding_model_dict['model']
        )
    else:
        embedding_function = OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=embedding_model_dict['model']
        )

    client = PersistentClient(path=collections_dataset_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={'hnsw:space': 'cosine'},
        embedding_function=embedding_function
    )

    llm_model = llm_model_dict['model']

    set_ids = set(ids)

    with tqdm(total=len(dataset['df']), file=sys.stdout, colour='blue',
              desc='\t\t\t  Running experiment') as pbar:

        for index, row in dataset['df'].iterrows():

            if row['id'] in set_ids:
                pbar.update(1)
                continue

            query = row['question']

            ids.append(row['id'])
            questions.append(query)
            answers.append(row['answer'])
            correct_contexts.append(row['context'])

            start = time.time()

            results = collection.query(
                query_texts=query,
                n_results=top_n
            )

            end = time.time()

            time_to_search_collection_model += end - start

            relevant_docs = results['documents'][0]

            relevant_docs = '\n'.join(relevant_docs)

            relevant_docs = relevant_docs.strip()

            relevant_contexts.append(relevant_docs)

            prompt_template = ChatPromptTemplate.from_template(prompt_template_base)

            prompt = prompt_template.format(contexto=relevant_docs, pergunta=query)

            start = time.time()

            response = llm_model.invoke(prompt)

            end = time.time()

            time_to_search_llm_model += end - start

            if llm_model_dict['cloud'] == 'local':
                question_answer = response.split('RESPOSTA:')[-1]
            else:
                question_answer = response.content

            question_answer = question_answer.replace('\n', ' ').strip()

            llm_answer.append(question_answer)

            pbar.update(1)

            time.sleep(time_sleep)

            llm_outputs_dict = {
                'llm_model_name': llm_model_dict['name'],
                'embedding_model_name': embedding_model_dict['name'],
                'dataset_name': dataset['url'],
                'time_to_search_collection_model': time_to_search_collection_model,
                'time_to_search_llm_model': time_to_search_llm_model,
                'ids': ids,
                'questions': questions,
                'answers': answers,
                'correct_contexts': correct_contexts,
                'relevant_contexts': relevant_contexts,
                'llm_answer': llm_answer
            }

            with open(file=output_file_path, mode='w', encoding='utf-8') as json_file:
                json.dump(llm_outputs_dict, json_file, indent=4)


