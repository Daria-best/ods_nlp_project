from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
)
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import Dataset
import pandas as pd
import numpy as np

from argparse import ArgumentParser, SUPPRESS
from tqdm import tqdm
import torch


# Set up command line execution
parser = ArgumentParser(
    add_help=False,
    description="Script to train models",
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
    help="path to the parquet file containing texts and queries ",
)


required.add_argument(
    "-n",
    "--model-name",
    required=True,
    type=str,
    help="choose one of the models: bge or stella",
)

required.add_argument(
    "-s",
    "--save-model-path",
    required=True,
    type=str,
    help="path for saving your trained model",
)

optional.add_argument(
    "-q",
    "--col-query",
    type=str,
    help="name of the column with queries in the sample",
    default="web_query_qwen",
)

optional.add_argument(
    "-c",
    "--col-corpus",
    type=str,
    help="name of the column with corpus in the sample",
    default="text_introduction",
)

args = parser.parse_args()


def create_corpus_queries_eval(df):
    corpus = {}
    queries = {}
    qrels = {}
    for i in tqdm(range(df.shape[0])):
        corpus[df.loc[i, "hash"] + "_c"] = df.loc[i, "text_introduction"]
        queries[df.loc[i, "hash"] + "_q"] = df.loc[i, "web_query_qwen"]
        if df.loc[i, "hash"] + "_q" not in qrels:
            qrels[df.loc[i, "hash"] + "_q"] = set()
        qrels[df.loc[i, "hash"] + "_q"].add(df.loc[i, "hash"] + "_c")
    return corpus, queries, qrels


def train_model(
    load_model_name,
    save_model_path,
    df,
    col_query,
    col_corpus,
    corpus_eval,
    queries_eval,
    qrels_eval,
):

    print(f"Loading model for training...")

    match load_model_name:
        case "bge":
            model = SentenceTransformer("BAAI/bge-large-en-v1.5").to("cuda")

        case "stella":
            model = SentenceTransformer(
                "dunzhang/stella_en_400M_v5", trust_remote_code=True
            ).cuda()
            model.default_prompt_name = "s2p_query"

        case _:
            print("you wrote wrong model name")
            return "error_model_name"

    anchors = list(df[col_query])
    positives = list(df[col_corpus])

    dataset = Dataset.from_dict(
        {
            "anchor": anchors,
            "positive": positives,
        }
    )

    loss_fn = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=save_model_path,
        num_train_epochs=1,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=500,
        learning_rate=0.00002,
        warmup_ratio=0.01,
        bf16=True,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=30,
        logging_steps=10,
    )

    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries_eval,
        corpus=corpus_eval,
        relevant_docs=qrels_eval,
        mrr_at_k=[10],
        ndcg_at_k=[5, 10],
        accuracy_at_k=[10],
        precision_recall_at_k=[10],
        map_at_k=[10],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss_fn,
        evaluator=ir_evaluator,
    )
    print("Training model...")
    trainer.train()

    print(f"Model was saved {save_model_path}")


print(f"Loading parquet file...")
df = pd.read_parquet(args.df_path)
df_eval = df[df.sample_df == "VALID"].reset_index(drop=True)
df_train = df[df.sample_df == "TRAIN"].reset_index(drop=True)
print(f"Loading eval sample...")
corpus_eval, queries_eval, qrels_eval = create_corpus_queries_eval(df_eval)

train_model(
    args.model_name,
    args.save_model_path,
    df,
    args.col_query,
    args.col_corpus,
    corpus_eval,
    queries_eval,
    qrels_eval,
)
