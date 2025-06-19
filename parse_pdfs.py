import pypdf
import re
import pandas as pd

from argparse import ArgumentParser, SUPPRESS
from tqdm import tqdm

# Set up command line execution
parser = ArgumentParser(
    add_help=False,
    description="Script for parsing NeurIPS papers from PDFs and cleaning the text",
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
    help="path to the Parquet file containing hashed paper paths",
)
required.add_argument(
    "-f", "--folder-path", required=True, type=str, help="folder with pdf path"
)
required.add_argument(
    "-m",
    "--meta-df-path",
    required=True,
    type=str,
    help="save path for the paper metadata DataFrame in .parquet format",
)

args = parser.parse_args()



def count_letters_and_others(s):
    letters = len(re.findall(r"[a-zA-Z]", s))
    others = len(s) - letters
    return letters, others


def clean_paper(text):
    pattern = r"[^a-zA-Z0-9\s.,!?:;\"'()\[\]{}\-–—«»]"
    text = re.sub(pattern, "", str(text))

    text = re.sub(r"\(cid:173\)\n", "", text)

    lines = text.split("\n")

    clean_lines = []
    for line in lines:
        letters, others = count_letters_and_others(line)
        if (
            (re.search(r"\b[a-zA-Z]{5,}\b", line))
            and (letters > others)
            and (line != "")
        ):
            clean_lines.append(line.strip())

    return clean_lines


def load_pdf_texts(df, path_folder):
    print(f"Parsing paper pdfs...")
    df["clean_text"] = ""
    df.reset_index(drop=True, inplace=True)
    for i in tqdm(range(df.shape[0])):
        try:
            reader = pypdf.PdfReader(f"{path_folder}/{df.loc[i, 'hash']}.pdf")
            text = ""
            for j in range(len(reader.pages)):
                text += reader.pages[j].extract_text()
            res = clean_paper(text)
            df.at[i, "clean_text"] = " ".join(res)
        except:
            df.loc[i, "clean_text"] = "NO TEXT"
    return df


print(f"Loading parquet file(s)...")
df = pd.read_parquet(args.df_path)
df = load_pdf_texts(df, args.folder_path)
print(
    f"{df[df.clean_text.str.len() >= 1000].shape[0]}/{df.shape[0]} pdfs, saving parser results..."
)
df[df.clean_text.str.len() >= 1000].reset_index(drop=True).to_parquet(args.meta_df_path)
