import os
import json

from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    answer_relevancy,
    context_recall,
    context_precision,
    faithfulness,
)


if __name__ == '__main__':

    load_dotenv()

    rag_outputs_dir = '../../data/rag/outputs/'

    rag_eval_results_dir = '../../data/results/rag_eval/'

    list_dataset_names = os.listdir(rag_outputs_dir)

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    llm_evaluator = LangchainLLMWrapper(
        ChatOpenAI(
            model='gpt-4o-mini',
            api_key=OPENAI_API_KEY
        )
    )

    embeddings_evaluator = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model='text-embedding-3-small',
            api_key=OPENAI_API_KEY
        )
    )

    print('\nRunning Evaluation')

    for cont_ds, dataset_name in enumerate(list_dataset_names, start=1):


        print(f'\n\tDataset {cont_ds}/{len(list_dataset_names)}: {dataset_name}')

        dataset_results_dir = os.path.join(rag_eval_results_dir, dataset_name)

        os.makedirs(dataset_results_dir, exist_ok=True)

        dataset_outputs_path = os.path.join(rag_outputs_dir, dataset_name)

        list_outputs_files = os.listdir(dataset_outputs_path)

        for cont, output_file_name in enumerate(list_outputs_files, start=1):

            print(f'\n\t\tOutput file {cont} / {len(list_outputs_files)}: {output_file_name}')

            if output_file_name == 'llama_31_405b_multilingual_e5_large_5.json':
                continue

            results_file_path = os.path.join(dataset_results_dir, output_file_name)

            if os.path.exists(results_file_path):
                continue

            output_file_path = os.path.join(dataset_outputs_path, output_file_name)

            with open(file=output_file_path, mode='r', encoding='utf-8') as json_file:
                outputs = json.load(json_file)



            ## Fracionando para evitar rate limit message error.

            tamanho_parte = len(outputs['questions']) // 10
            for i in range(10):
                inicio = i * tamanho_parte
                fim = (i + 1) * tamanho_parte

                sample = {
                    'question': outputs['questions'][inicio:fim],
                    'ground_truth': outputs['answers'][inicio:fim],
                    'answer': [x.strip() for x in outputs['llm_answer']][inicio:fim],
                    'contexts': [x.split('\n\n') for x in outputs['relevant_contexts']][inicio:fim]
                }

                eval_dataset = Dataset.from_dict(sample)

                scores = evaluate(
                    dataset=eval_dataset,
                    llm=llm_evaluator,
                    embeddings=embeddings_evaluator,
                    metrics=[
                        # context_recall,
                        # context_precision,
                        answer_relevancy,
                        faithfulness,
                    ],
                    run_config=RunConfig(max_workers=3)
                )

                results_dataframe = scores.to_pandas()

                results_json = results_dataframe.to_json(orient='records')

                results_json_parsed = json.loads(results_json)

                with open(file=results_file_path, mode='w', encoding='utf-8') as json_file:
                    json.dump(results_json_parsed, json_file, indent=4)







            # sample = {
            #     'question': outputs['questions'],
            #     'ground_truth': outputs['answers'],
            #     'answer': [x.strip() for x in outputs['llm_answer']],
            #     'contexts': [x.split('\n\n') for x in outputs['relevant_contexts']]
            # }

            # eval_dataset = Dataset.from_dict(sample)

            # scores = evaluate(
            #     dataset=eval_dataset,
            #     llm=llm_evaluator,
            #     embeddings=embeddings_evaluator,
            #     metrics=[
            #         # context_recall,
            #         # context_precision,
            #         answer_relevancy,
            #         faithfulness,
            #     ],
            #     run_config=RunConfig(max_workers=3)
            # )
            #
            # results_dataframe = scores.to_pandas()
            #
            # results_json = results_dataframe.to_json(orient='records')
            #
            # results_json_parsed = json.loads(results_json)
            #
            # with open(file=results_file_path, mode='w', encoding='utf-8') as json_file:
            #     json.dump(results_json_parsed, json_file, indent=4)
