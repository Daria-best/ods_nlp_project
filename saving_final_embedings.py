from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
from argparse import ArgumentParser, SUPPRESS
from tqdm import tqdm
import torch


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
    "-m",
    "--meta-df-path",
    required=True,
    type=str,
    help="path to save paper embeddings in .parquet format",
)

optional.add_argument(
    "-p",
    "--path-model",
    type=str,
    help="load model from your computer",
    default="Daria-best/stella_en_400M_v5_neurlips_papers_fine-tuned",
)

optional.add_argument(
    "-son",
    "--save_only_nesessery",
    type=str,
    help="save only necessary columns for paper search: yes or no",
    default="yes",
)

args = parser.parse_args()


def embeding_corpus(df, path_model):
    df = df.reset_index(drop=True)
    df["paper_embeding"] = None
    model = SentenceTransformer(path_model, trust_remote_code=True).to("cuda")
    for i in tqdm(range(df.shape[0])):
        with torch.no_grad():
            outputs = model.encode(
                df.loc[i, "clean_text"],
                convert_to_tensor=True,
                convert_to_numpy=False,
                device="cuda",
                prompt_name="s2p_query",
            )
        df.at[i, "paper_embeding"] = np.array(outputs.cpu())
    return df


print(f"Loading parquet file...")
df = pd.read_parquet(args.df_path)
print(f"Create embedings...")
df = embeding_corpus(df, args.path_model)
if args.save_only_nesessery == "yes":
    df = df[["title", "abstract", "year", "paper_embeding"]]
df.to_parquet(args.meta_df_path)
print(f"File was saved in {args.meta_df_path}")
