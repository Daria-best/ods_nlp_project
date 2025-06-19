import pypdf
import re
import pandas as pd
import numpy as np

from argparse import ArgumentParser, SUPPRESS
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import torch
import faiss

# Set up command line execution
parser = ArgumentParser(
    add_help=False,
    description="Script to find NeurlIPS papers matching your request",
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
    "-e",
    "--path-embedings-parquet",
    required=True,
    type=str,
    help="path to the parquet file containing titles, abstracts, and paper embeddings",
)
required.add_argument("-m", "--path-model", required=True, type=str, help="model path")

args = parser.parse_args()


def get_top_papers(df_embedings, query, model, faiss_index):

    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)
    distances, indices = faiss_index.search(query_vector, 10)

    for i, index in enumerate(indices[0]):
        distance = distances[0][i]
        print(
            f"TOP {i+1} paper:\nTITLE: {df_embedings.loc[index, 'title']} {df_embedings.loc[index, 'year']} \nABSTRACT: {df_embedings.loc[index, 'abstract']}"
        )


def main():
    print("Loading embedings...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_embedings = pd.read_parquet(args.path_embedings_parquet)

    document_embeddings = np.array(list(df_embedings.paper_embeding), dtype=np.float32)
    dim = len(document_embeddings[0])
    faiss_index = faiss.IndexFlatIP(dim)  
    
    faiss.normalize_L2(document_embeddings)
    faiss_index.add(document_embeddings)

    model = SentenceTransformer(args.path_model, trust_remote_code=True).to(device)

    print("The program is running. Enter your query.")
    print("To exit, type 'exit'.\n")

    while True:
        user_input = input("Запрос: ")

        if user_input.lower() in ["exit", "EXIT", "Exit"]:
            print("The program is completed.")
            break

        if not user_input:
            print("Empty input. Please try again.\n")
            continue

        get_top_papers(df_embedings, user_input, model, faiss_index)


if __name__ == "__main__":
    main()
