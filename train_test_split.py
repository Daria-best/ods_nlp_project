import pandas as pd
import numpy as np

from argparse import ArgumentParser, SUPPRESS
from tqdm import tqdm


# Set up command line execution
parser = ArgumentParser(
    add_help=False,
    description="Script to perform train-test-validation split",
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
    "--path-parquet-df",
    required=True,
    type=str,
    help="path to the parquet file containing titles, abstracts, and paper embeddings",
)

required.add_argument(
    "-m",
    "--meta-df-path",
    required=True,
    type=str,
    help="save path for the paper metadata dataframe in .parquet format",
)

args = parser.parse_args()


def cut_before_introduction(text):
    keyword = "introduction"
    idx = text.lower().find(keyword)
    if idx > -1:
        return idx, text[idx:]
    else:
        return idx, "introduction not found"


print(f"Loading parquet file(s)...")
df = pd.read_parquet(args.path_parquet_df)
print(f"Cut text from 'introduction'...")
df["introd_index"], df["text_introduction"] = zip(
    *df.clean_text.apply(lambda x: cut_before_introduction(str(x)))
)
print(f"Drop papers without 'introduction'")
df = df[df.introd_index > -1]

df = df.sort_values(["year", "title"]).reset_index(drop=True)
np.random.seed(17)
list_index = np.random.permutation(df.shape[0])
list_index_test = list_index[:5000]
list_index_valid = list_index[5000:10000]

print(f"Split train-test-validation")
df.loc[df.index.isin(list_index_test), "sample_df"] = "TEST"
df.loc[df.index.isin(list_index_valid), "sample_df"] = "VALID"
df.loc[df.sample_df.isna(), "sample_df"] = "TRAIN"

df.to_parquet(args.meta_df_path)
print(f"File was saved in {args.meta_df_path}")
