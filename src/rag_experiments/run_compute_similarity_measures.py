import os
import json
import pandas as pd

from evaluate import load
from bert_score import BERTScorer


def compute_bertscore(list_reference_answer_, list_generated_answer_):
    scorer = BERTScorer(lang='pt', rescale_with_baseline=False)
    precision, recall, f1_score = scorer.score(list_generated_answer_, list_reference_answer_, verbose=True)
    return {
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1_score': f1_score.mean().item()
    }


def compute_rouge(list_reference_answer_, list_generated_answer_):
    metric = load('rouge')
    results = metric.compute(predictions=list_generated_answer_, references=list_reference_answer_,
                             use_stemmer=False)
    return results


if __name__ == '__main__':

    rag_outputs_dir = '../../data/rag/outputs/'


    rag_eval_results_dir = '../../data/results/rag_eval_sim/'

    os.makedirs(rag_eval_results_dir, exist_ok=True)

    list_dataset_names = os.listdir(rag_outputs_dir)

    print('\nRunning Evaluation')

    for cont_ds, dataset_name in enumerate(list_dataset_names, start=1):

        print(f'\n\tDataset {cont_ds}/{len(list_dataset_names)}: {dataset_name}')

        dataset_outputs_path = os.path.join(rag_outputs_dir, dataset_name)

        list_outputs_files = os.listdir(dataset_outputs_path)

        list_results = []

        for cont, output_file_name in enumerate(list_outputs_files, start=1):

            print(f'\n\t\tOutput file {cont} / {len(list_outputs_files)}: {output_file_name}')

            output_file_path = os.path.join(dataset_outputs_path, output_file_name)

            with open(file=output_file_path, mode='r', encoding='utf-8') as json_file:
                outputs = json.load(json_file)

            list_llm_answer = outputs['llm_answer']
            list_ground_truth = outputs['answers']



            bert_score = compute_bertscore(list_ground_truth, list_llm_answer)

            rouge_metrics = compute_rouge(list_ground_truth, list_llm_answer)

            # print(bert_score)
            #
            # print(rouge_metrics)

            model_name = output_file_name.replace('.json', '')

            dict_results = {
                'model_name': model_name,
                'bert_score_recall': bert_score['recall'],
                'bert_score_precision': bert_score['precision'],
                'bert_score_f1': bert_score['f1_score'],
                'rouge_1': rouge_metrics['rouge1'],
                'rouge_2': rouge_metrics['rouge2'],
                'rouge_l': rouge_metrics['rougeL'],
            }

            list_results.append(dict_results)

        df = pd.DataFrame(list_results)

        results_file_path = os.path.join(rag_eval_results_dir, f'results_{dataset_name}.csv')

        df.to_csv(results_file_path)
