import torch
import numpy as np
from tqdm import tqdm


class MY_AUTO_MODEL:
    def __init__(
        self,
        embedding_model,
        tokenizer,
        encoder_batch_size,
        use_detailed_instruct,
        use_quer_maxlength,
        max_length,
        use_normalize,
        **kwargs,
    ):
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.encoder_batch_size = encoder_batch_size
        self.use_detailed_instruct = use_detailed_instruct
        self.use_quer_maxlength = use_quer_maxlength
        self.max_length = max_length
        self.use_normalize = use_normalize

    def _split_list(self, input_list, chunk_size):
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i : i + chunk_size]

    def _last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def _get_detailed_instruct(self, query):
        return f"Instruct: Given a search query, retrieve papers that contain the most relevant information\nQuery: {query}"

    def encode_queries(
        self, queries, batch_size, **kwargs
    ):  # тут queries передается как лист со строками (строка=текст)
        total_encoded_queries = []
        print("encoded_queries")
        for query_chunks in tqdm(
            self._split_list(queries, self.encoder_batch_size)
        ):  # берется encoder_batch_size строк
            if self.use_detailed_instruct:
                query_chunks = [self._get_detailed_instruct(t) for t in query_chunks]

            if self.use_quer_maxlength:
                inputs = self.tokenizer(
                    query_chunks,
                    return_tensors="pt",
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                ).to(self.embedding_model.device)
            else:
                inputs = self.tokenizer(
                    query_chunks, return_tensors="pt", padding=True
                ).to(self.embedding_model.device)

            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            if self.use_normalize:
                sentence_embeddings = outputs[0][:, 0]
                total_encoded_queries += torch.nn.functional.normalize(
                    sentence_embeddings, p=2, dim=1
                ).cpu()
            else:
                total_encoded_queries += self._last_token_pool(
                    outputs.last_hidden_state, inputs["attention_mask"]
                ).cpu()
            del inputs
            del outputs
        return np.array(total_encoded_queries)

    def encode_corpus(self, corpus, batch_size, **kwargs):
        passages = [passage["title"] + " " + passage["text"] for passage in corpus]
        total_passages_corpus = []
        print("encoded_corpus")
        for passages_chunks in tqdm(
            self._split_list(passages, self.encoder_batch_size)
        ):
            inputs = self.tokenizer(
                passages_chunks,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            ).to(self.embedding_model.device)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            if self.use_normalize:
                sentence_embeddings = outputs[0][:, 0]
                total_passages_corpus += torch.nn.functional.normalize(
                    sentence_embeddings, p=2, dim=1
                ).cpu()
            else:
                total_passages_corpus += self._last_token_pool(
                    outputs.last_hidden_state, inputs["attention_mask"]
                ).cpu()
            del inputs
            del outputs
        return np.array(total_passages_corpus)


class MY_SENTENCE_TRANSFORMER_MODEL:
    def __init__(self, embedding_model, encoder_batch_size, **kwargs):
        self.embedding_model = embedding_model
        self.encoder_batch_size = encoder_batch_size

    def _split_list(self, input_list, chunk_size):
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i : i + chunk_size]

    def encode_queries(
        self, queries, batch_size, **kwargs
    ):  # тут queries передается как лист со строками (строка=текст)
        total_encoded_queries = []
        print("encoded_queries")
        for query_chunks in tqdm(
            self._split_list(queries, self.encoder_batch_size)
        ):  # берется encoder_batch_size строк
            with torch.no_grad():
                outputs = self.embedding_model.encode(
                    query_chunks,
                    convert_to_tensor=True,
                    convert_to_numpy=False,
                    device="cuda",
                    prompt_name="s2p_query",
                )
            total_encoded_queries += outputs.cpu()
            del outputs
        return np.array(total_encoded_queries)

    def encode_corpus(self, corpus, batch_size, **kwargs):
        passages = [passage["title"] + " " + passage["text"] for passage in corpus]
        total_passages_queries = []
        print("encoded_corpus")
        for passages_chunks in tqdm(
            self._split_list(passages, self.encoder_batch_size)
        ):
            with torch.no_grad():
                outputs = self.embedding_model.encode(
                    passages_chunks,
                    convert_to_tensor=True,
                    convert_to_numpy=False,
                    device="cuda",
                    prompt_name="s2p_query",
                )
            total_passages_queries += outputs.cpu()
            del outputs
        return np.array(total_passages_queries)
