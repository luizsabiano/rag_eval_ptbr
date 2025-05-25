# <p style="text-align:center;"> Comparação de Modelos de Embeddings e LLMs para Geração Aumentada por Recuperação em Português</p> 

---
## Resumo do artigo

<p style="text-align:justify;">Os modelos de linguagem de larga escala (LLMs) representam um avanço para a área de processamento de linguagem natural, impulsionando o desempenho em tarefas como geração de texto e resposta a perguntas. No entanto, eles enfrentam desafios como alucinações e falta de acesso a informações atualizadas. A técnica de geração aumentada por recuperação (RAG) busca mitigar esses problemas ao integrar recuperação de informações externas à geração de texto, melhorando a precisão e a atualidade das respostas. Este trabalho realizou uma investigação de diversos modelos embeddings e LLMs de código aberto e proprietários aplicados à técnica RAG considerando três bases de dados contendo documentos escritos em português do Brasil. Os resultados experimentais demonstraram que os modelos Multilingual E5 large e Gemma 2 9B obtiveram o melhor desempenho dentre os modelos avaliados com base em diferentes medidas de avaliação.</p>

Autores: Luiz Sabiano F. Medeiros e Hilário Tomaz Alvez de Oliveira

Acesse o artigo completo em [CSBC](https://teste.com.br/teste)

---

## Estrutura de diretórios


<pre>├── <font color="#12488B"><b>data</b></font>
│   ├── <font color="#12488B"><b>collections</b></font> <font color="#7B68EE"> => Pasta que armazena as coleções de embeddings</font>
│   ├── <font color="#12488B"><b>rag</b></font>
│   │   └── <font color="#12488B"><b>outputs</b></font>
│   ├── <font color="#12488B"><b>resources</b></font>  
│   │   └── prompt.txt <font color="#7B68EE">==> Prompt submetido ao LLM</font>
│   └── <font color="#12488B"><b>results</b></font> 
│       ├── <font color="#12488B"><b>embeddings_eval</b></font> <font color="#7B68EE"> ==> Armazena json com dados recuperados pelo modelo de embedding e cvs do score encontrado</font>
│       ├── <font color="#12488B"><b>rag_eval</b></font>
│       └── <font color="#12488B"><b>rag_eval_sim</b></font>
├── <font color="#12488B"><b>src</b></font>
│   ├── <font color="#12488B"><b>chromadb</b></font>
│   │   └── chromadb_functions.py
│   ├── <font color="#12488B"><b>datasets</b></font>
│   │   ├── local_dataset.py <font color="#7B68EE"> ==> Datasets</font>
│   ├── <font color="#12488B"><b>embedding_models</b></font>
│   │   ├── embedding_models.py <font color="#7B68EE"> ==> Modelos de embeddings</font>
│   ├── <font color="#12488B"><b>embeddings_experiments</b></font> <font color="#7B68EE">  ==> Experimentos 1 - Modelos de Embeddins</font>
│   │   ├── run_embedding_models_eval.py </font> <font color="#7B68EE"> ==> Executável </font>
│   │   └── run_summarize_emb_results.py <font color="#7B68EE">  ==> Executável </font>
│   └── <font color="#12488B"><b>rag_experiments</b></font> <font color="#7B68EE">  ==> Experimentos 2 - RAG</font>
│       ├── rag_utils.py
│       ├── run_compute_similarity_measures.py <font color="#7B68EE">  ==> Executável</font>
│       ├── run_rag_eval.py  <font color="#7B68EE">  ==> Executável</font>
│       ├── run_rag_generate_outputs_cloud.py <font color="#7B68EE">  ==> Executável</font>
│       └── run_summarize_rag_results.py <font color="#7B68EE">  ==> Executável</font>
├── README.md
└── requirements.txt <font color="#7B68EE">O arquivo de requisitos para reproduzir os experimentos</font>
</pre>


### Instalação das dependências: 
$ pip install -i requirements.txt


## Experimento 1 - Avaliação dos Modelos de Embedding

1) run_embedding_models_eval.py: carrega os datasets e os amazena no banco de dados de embeddings (ChromaDB) na pasta data/collections/dataset_name.
Para cada pergunta, realiza a busca semantica, retorna n passagens e os salva em results/embeddings_eval/dataset_name. 
2) run_summarize_emb_results.py: extrai as informações do json salvo anteriormente e calcula as estatística de acerto (recall). 

## Experimento 2 - Avaliação da técnica RAG 

Necessário GPU Nvidia.

1) run_rag_generate_outputs_local.py: RAG aplicada aos LLMs locais, resultado salvo em data/rag/outputs/dataset_name
2) run_rag_generate_outputs_cloud.py: RAG aplicada aos LLMs em nuvem, resultado salvo em data/rag/outputs/dataset_name
3) run_compute_similarity_measures.py: Calcula as métricas BertScore e Rouge, comparando respostas dos LLMs as respostas de referência. Salva resultados em ../../data/results/rag_eval_sim/
4) run_rag_eval.py: Avalia as respostas retornadadas pelos LLMs com o Framework Ragas. Salva em ../../data/results/rag_eval/dataset_name
5) run_summarize_rag_results.py: extrai as informações do json salvo anteriormente e calcula as estatística computadas através pelo RAGAS.