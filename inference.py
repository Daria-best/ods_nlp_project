import my_models_class
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
import pandas as pd
import numpy as np

from argparse import ArgumentParser, SUPPRESS
from tqdm import tqdm
import torch


# Set up command line execution
parser = ArgumentParser(
    add_help=False,
    description="Script to load models and compute NDCG results",
)

required = parser.add_argument_group("required arguments")
optional = parser.add_argument_group("optional arguments")

optional.add_argument(
    "-h",
    "--help",
    action="help",
    default=SUPPRESS,
    help="show this help message and exit",
)

required.add_argument(
    "-d",
    "--df-path",
    required=True,
    type=str,
    help="path to the parquet file containing texts and queries",
)


required.add_argument(
    "-n",
    "--model-name",
    required=True,
    type=str,
    help="choose one of the models: bge, e5_mistral, linq, stella",
)

optional.add_argument(
    "-t",
    "--text-split",
    type=str,
    help="hoose text split: text_full or text_introduction",
    default="text_introduction",
)

optional.add_argument(
    "-q",
    "--col-query",
    type=str,
    help="name the column with queries in the sample",
    default="web_query_qwen",
)

optional.add_argument(
    "-l",
    "--path-to-your-local-model",
    type=str,
    help="load model from your computer",
    default="no_data",
)


args = parser.parse_args()


def create_corpus_queries(df, col_query, text_split):
    df = df.reset_index(drop=True)
    corpus = {}
    queries = {}
    qrels = {}
    for i in tqdm(range(df.shape[0])):
        if text_split == "text_full":
            corpus[df.loc[i, "hash"] + "_c"] = {
                "text": df.loc[i, "clean_text"],
                "title": "",
            }
        else:
            corpus[df.loc[i, "hash"] + "_c"] = {
                "text": df.loc[i, "text_introduction"],
                "title": "",
            }
        queries[df.loc[i, "hash"] + "_q"] = df.loc[i, col_query]
        qrels[df.loc[i, "hash"] + "_q"] = {df.loc[i, "hash"] + "_c": 1}
    return corpus, queries, qrels


def get_model_result(load_model_name, model_path, corpus, queries, qrels):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # до какого размерности сжимается входной объект
        bnb_4bit_compute_dtype=torch.bfloat16,  # до какого размера сжимается сетка
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    match load_model_name:
        case "bge":
            if model_path == "no_data":
                model_name = "BAAI/bge-large-en-v1.5"
            else:
                model_name = model_path
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, device_map="cuda:0", tranketion_strategy="only_first"
            )
            model = AutoModel.from_pretrained(model_name, device_map="cuda:0")
            model_from_class = my_models_class.MY_AUTO_MODEL(
                model, tokenizer, 1028, False, False, 512, True
            )
            model_dres = DRES(model_from_class, batch_size=1028)

        case "e5_mistral":
            model_name = "intfloat/e5-mistral-7b-instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda:0")
            model = AutoModel.from_pretrained(
                model_name, quantization_config=bnb_config, device_map="cuda:0"
            )
            model_from_class = my_models_class.MY_AUTO_MODEL(
                model, tokenizer, 64, False, False, 4096, False
            )
            model_dres = DRES(model_from_class, batch_size=64)

        case "linq":
            model_name = "Linq-AI-Research/Linq-Embed-Mistral"
            tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda:0")
            model = AutoModel.from_pretrained(
                model_name, quantization_config=bnb_config, device_map="cuda:0"
            )
            model_from_class = my_models_class.MY_AUTO_MODEL(
                model, tokenizer, 16, True, True, 4096, False
            )
            model_dres = DRES(model_from_class, batch_size=16)

        case "stella":
            if model_path == "no_data":
                model_name = "dunzhang/stella_en_400M_v5"
            else:
                model_name = model_path
            model = SentenceTransformer(model_name, trust_remote_code=True).cuda()
            model_from_class = my_models_class.MY_SENTENCE_TRANSFORMER_MODEL(
                model, 1028
            )
            model_dres = DRES(model_from_class, batch_size=1028)

        case _:
            print("you wrote wrong model name")
            return "error_model_name"

    retriever = EvaluateRetrieval(model_dres, score_function="cos_sim")
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results, retriever.k_values
    )
    return ndcg, recall, precision


print(f"Loading parquet file...")
df = pd.read_parquet(args.df_path)
df_test = df[df.sample_df == "TEST"].reset_index(drop=True)
corpus_test, queries_test, qrels_test = create_corpus_queries(
    df_test, args.col_query, args.text_split
)
print(f"Calculation of the model on the test sample...")
res = get_model_result(
    args.model_name,
    args.path_to_your_local_model,
    corpus_test,
    queries_test,
    qrels_test,
)
if res != "error_model_name":
    print("NDCG_1", res[0]["NDCG@1"])
    print("NDCG_5", res[0]["NDCG@5"])
    print("NDCG_10", res[0]["NDCG@10"])
