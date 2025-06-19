import pypdf
import pandas as pd
import numpy as np

from argparse import ArgumentParser, SUPPRESS
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    T5Tokenizer,
    T5ForConditionalGeneration,
    pipeline,
)
import torch


# Set up command line execution
parser = ArgumentParser(
    add_help=False,
    description="Script to generate paper requests",
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
    help="path to the parquet file containing abstracts",
)


required.add_argument(
    "-n",
    "--model-name",
    required=True,
    type=str,
    help="choose one of the models: zephyr, google, mathstral, qwen",
)

required.add_argument(
    "-m",
    "--meta-df-path",
    required=True,
    type=str,
    help="save path for the paper metadata dataframe in .parquet format",
)

args = parser.parse_args()


def load_queries(df, model_name):
    df = df.reset_index(drop=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # до какого размерности сжимается входной объект
        bnb_4bit_compute_dtype=torch.bfloat16,  # до какого размера сжимается сетка
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    match model_name:
        case "zephyr":
            tokenizer_zeph = AutoTokenizer.from_pretrained(
                "HuggingFaceH4/zephyr-7b-beta", device_map="cuda:0"
            )
            model_zeph = AutoModelForCausalLM.from_pretrained(
                "HuggingFaceH4/zephyr-7b-beta",
                quantization_config=bnb_config,
                device_map="cuda:0",
            )

            for i in tqdm(range(df.shape[0])):
                abstract = df["abstract"][i]
                prompt = (
                    "<|system|>\n"
                    "You are a helpful assistant that extracts general search queries from academic abstracts. "
                    "Lenght of queries must be less seven words"
                    "Avoid names or terms introduced in the paper. Use only general terms commonly used in machine learning and data science.\n"
                    "<|user|>\n"
                    "Summarize the following abstract into a short, general phrase of 7-10 words suitable for a web search query. "
                    "Lenght of queries must be less seven words"
                    "Avoid or terms introduced in the paper. Only output the search query.\n\n"
                    "Example1:\n"
                    "Abstract: Graph neural networks (GNNs) have been successful in learning representations from graphs. Many popular GNNs follow the pattern of aggregate-transform: they aggregate the neighbors attributes and then transform the results of aggregation with a learnable function. Analyses of these GNNs explain which pairs of non-identical graphs have different representations. However, we still lack an understanding of how similar these representations will be. We adopt kernel distance and propose transform-sum-cat as an alternative to aggregate-transform to reflect the continuous similarity between the node neighborhoods in the neighborhood aggregation. The idea leads to a simple and efficient graph similarity, which we name Weisfeiler-Leman similarity (WLS). In contrast to existing graph kernels, WLS is easy to implement with common deep learning frameworks. In graph classification experiments, transform-sum-cat significantly outperforms other neighborhood aggregation methods from popular GNN models. We also develop a simple and fast GNN model based on transform-sum-cat, which obtains, in comparison with widely used GNN models, (1) a higher accuracy in node classification, (2) a lower absolute error in graph regression, and (3) greater stability in adversarial training of graph generation.\n"
                    "Search query: graph neural network similarity kernel distance\n\n"
                    "Example2:\n"
                    f"Abstract:\n{abstract}\n\n"
                    "Search query:\n"
                    "<|assistant|>\n"
                )

                inputs = tokenizer_zeph(prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model_zeph.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pad_token_id=tokenizer_zeph.pad_token_id,
                    )
                df.loc[i, f"web_query_{model_name}"] = (
                    tokenizer_zeph.decode(outputs[0])
                    .split("<|assistant|>")[-1]
                    .split("Example2:")[-1]
                    .strip()
                )

        case "google":
            tokenizer_goog = T5Tokenizer.from_pretrained(
                "google/flan-t5-base", device_map="cuda:0"
            )
            model_goog = T5ForConditionalGeneration.from_pretrained(
                "google/flan-t5-base", device_map="cuda:0"
            )

            for i in tqdm(range(df.shape[0])):  # df_check.shape[0]
                abstract = df["abstract"][i]
                promt = f"You are a researcher writing a web search query to find papers similar to the one below. Write a short, \
                            general search phrase that uses only common terms in the field and could be typed by someone who hasn't read the paper. \
                            Avoid using specific method names, acronyms, or terminology introduced in the abstract.\
                            Avoid specific names, methods, or technical terms introduced in the text.\
                            Abstract: {abstract}\
                            Search quuery:"

                input_ids = tokenizer_goog(promt, return_tensors="pt").input_ids.to(
                    "cuda"
                )
                with torch.no_grad():
                    outputs = model_goog.generate(input_ids, temperature=1)
                df.loc[i, f"web_query_{model_name}"] = tokenizer_goog.decode(outputs[0])

        case "mathstral":
            tokenizer_math = AutoTokenizer.from_pretrained(
                "mistralai/Mathstral-7b-v0.1", device_map="cuda"
            )
            model_math = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mathstral-7b-v0.1",
                device_map="cuda",
                quantization_config=bnb_config,
            )
            generator = pipeline(
                "text-generation",
                model=model_math,
                tokenizer=tokenizer_math,
                do_sample=True,
                temperature=1,
                top_p=0.95,
                pad_token_id=tokenizer_math.eos_token_id,
            )

            for i in tqdm(range(df.shape[0])):
                abstract = df["abstract"][i]
                promt = f"Given the following abstract of a scientific paper, generate a short and general web search query that\
                could help someone find this paper. Do not use specific names, acronyms, or terms introduced in the paper. \
                Use general terms common in the field.\
                Abstract:{abstract}\
                Search query:"

                with torch.no_grad():
                    out = generator(promt)
                df.loc[i, f"web_query_{model_name}"] = out[0]["generated_text"].split(
                    "Search query:"
                )[-1]

        case "qwen":
            model_qwen = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-7B-Instruct",
                quantization_config=bnb_config,
                device_map="cuda",
            )
            tokenizer_qwen = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2-7B-Instruct", device_map="cuda"
            )
            for i in tqdm(range(df.shape[0])):
                abstract = df["abstract"][i]

                messages = [
                    {
                        "role": "user",
                        "content": (
                            "Given the following abstract, generate a short, general search query (a few words only) that captures its key contribution. "
                            "Avoid repeating the paper title or using dataset- or model-specific names. "
                            "Do not list keywords. Instead, return a single coherent phrase or question that someone might use in a web search to find this work. "
                            "The phrase should use only general academic or technical terminology, and be under 8 words.\n\n"
                            f"Abstract: {abstract}\n\n"
                            "Search query:"
                        ),
                    }
                ]
                text = tokenizer_qwen.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                model_inputs = tokenizer_qwen([text], return_tensors="pt").to("cuda")

                with torch.no_grad():
                    generated_ids = model_qwen.generate(
                        model_inputs.input_ids,
                        attention_mask=model_inputs.attention_mask,
                        temperature=1,
                        do_sample=True,
                    )
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        model_inputs.input_ids, generated_ids
                    )
                ]

                df.loc[i, f"web_query_{model_name}"] = tokenizer_qwen.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
            df["web_query_qwen"] = df.web_query_qwen.str.strip('"').str.strip()

        case _:
            print("model not found! file will be saved without any changes")
    return df


print(f"Loading parquet file...")
df = pd.read_parquet(args.df_path)
print(f"Generate requests...")
df = load_queries(df, args.model_name)
df.to_parquet(args.meta_df_path)
print(f"File was saved in {args.meta_df_path}")
