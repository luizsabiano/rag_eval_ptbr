from datasets import load_dataset


# Lista de datasets utilizados
URL_DATASETS = [
    'paulopirozelli/pira',
    'benjleite/FairytaleQA-translated-ptBR',
    'tiagofvb/squad2-pt-br-no-impossible-questions',
]


# Descarta as colunas não alvo e exclui ids duplicados
def load_important_columns(df):
    # keep first duplicate value
    df = df.drop_duplicates(subset=['id'])
    df = df[['id', 'question', 'answer', 'context']]
    return df


# Converge o nome das colunas nos diferentes dataset para um padrão único.
def set_the_same_structure(url, df):
    if url == 'pira':
        df = df.rename(
            columns={'id_qa': 'id', 'question_pt_origin': 'question',
                     'answer_pt_origin': 'answer', 'abstract_translated_pt': 'context'
                     }
        )
    elif url == 'FairytaleQA-translated-ptBR':
        df = df.rename(columns={'story_section': 'context'})
        indice = []
        for i in range(df.shape[0]):
            # indice.append(hex(i))
            indice.append(str(i))
        df['id'] = indice
    return df.copy()


# Recebe o url dos datasets, baixa, normaliza
def load_datasets_base(urls: list, dataset_split: str = 'test'):

    df_list = []

    for url in urls:

        dataset = load_dataset(url, 'default')

        # Seleciona versao para trabalho, opções: test ou train
        df = dataset[dataset_split].to_pandas()

        dataset_name = url.split('/')[1]

        df = set_the_same_structure(dataset_name, df)

        df = load_important_columns(df)

        # limitação para teste de código
        # Em produção retirar
        # df = df.head(30)

        df_list.append({'url': dataset_name, 'df': df})


    return df_list


def get_dataframe_content(dataframe):
    ids = []
    documents = []
    df = dataframe.reset_index()
    df = df.drop_duplicates(subset=['context'])
    for index, row in df.iterrows():
        ids.append(row['id'])
        documents.append(row['context'])
    return ids, documents
