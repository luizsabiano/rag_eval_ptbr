import torch

from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer


EMBEDDING_MODELS = [

    # {
    #     'name': 'msmarco_distilbert_100k',
    #     'ef': 'LocalHuggingFaceEmbeddingFunction',
    #     'model': 'mpjan/msmarco-distilbert-base-tas-b-mmarco-pt-100k'
    # },
    #
    # {
    #     'name': 'paraphrase_minilm_l12_v2',
    #     'ef': 'LocalHuggingFaceEmbeddingFunction',
    #     'model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    # },
    #
    # {
    #      'name': 'paraphrase_multilingual',
    #      'ef': 'LocalHuggingFaceEmbeddingFunction',
    #      'model': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    #  },
    #
    # {
    #     'name': 'vabatista_sbert_bm25',
    #     'ef': 'LocalHuggingFaceEmbeddingFunction',
    #     "model": 'vabatista/sbert-mpnet-base-bm25-hard-neg-pt-br'
    # },
    #
    # {
    #     'name': 'utl5_small',
    #     'ef': 'LocalHuggingFaceEmbeddingFunction',
    #     'model': 'tgsc/sentence-transformer-ult5-pt-small'
    # },

    # {
    #     'name': 'multilingual_e5_small',
    #     'ef': 'LocalHuggingFaceEmbeddingFunction',
    #     'model': 'intfloat/multilingual-e5-small'
    # },
    #
    # {
    #     'name': 'multilingual_e5_base',
    #     'ef': 'LocalHuggingFaceEmbeddingFunction',
    #     'model': 'intfloat/multilingual-e5-base'
    # },

    {
        'name': 'multilingual_e5_large',
        'ef': 'LocalHuggingFaceEmbeddingFunction',
        'model': 'intfloat/multilingual-e5-large'
    },

    # {
    #     'name': 'Portulan_serafim_100m',
    #     'ef': 'LocalHuggingFaceEmbeddingFunction',
    #     'model': 'PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir'
    # },
    #
    # {
    #     'name': 'Portulan_serafim_900m',
    #     'ef': 'LocalHuggingFaceEmbeddingFunction',
    #     'model': 'PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir'
    # },
    #
    # {
    #     'name': 'OpenAI_3_small',
    #     'ef': 'OpenAIEmbeddingFunction',
    #     'model': 'text-embedding-3-small'
    # },
    #
    # {
    #     'name': 'OpenAI_3_large',
    #     'ef': 'OpenAIEmbeddingFunction',
    #     'model': 'text-embedding-3-large'
    # }

]


class LocalHuggingFaceEmbeddingFunction(EmbeddingFunction[Documents]):

    def __init__(self, model_name: str):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        torch.cuda.empty_cache()

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()
