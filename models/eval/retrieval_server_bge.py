# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py

import argparse
import warnings
from typing import Optional

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BGE-M3 model on device: {device}")
    model = SentenceTransformer(
        model_path,
        device=device,
        revision="babcf60cae0a1f438d7ade582983d4ba462303c2",
    )
    model.eval()
    print("Loaded BGE-M3 model.")
    return model


class Encoder:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    @torch.no_grad()
    def encode(self, query_list: list[str]) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        query_emb = self.model.encode(
            sentences=query_list,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        query_emb = query_emb.astype(np.float32, order="C")
        torch.cuda.empty_cache()
        return query_emb


class DenseRetriever:
    def __init__(self, config):
        self.config = config
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

        self.index = faiss.read_index(config.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(config.corpus_path)
        self.encoder = Encoder(model_path=config.retrieval_model_path)

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        results = load_docs(self.corpus, idxs[0])
        if return_score:
            return results, scores[0].tolist()
        return results

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Retrieval process: ", disable=len(query_list) < 20):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            results.extend(batch_results)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()

        if return_score:
            return results, scores
        return results

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


#####################################
# FastAPI server below
#####################################


class Config:
    def __init__(
        self,
        retrieval_topk: int = 3,
        index_path: str = "./retriever_index/bge_m3_Flat.index",
        corpus_path: str = "./retriever_index/wiki-24.jsonl",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "BAAI/bge-m3",
        retrieval_batch_size: int = 512,
    ):
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.

    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }

    Output format (when return_scores=True, similarity scores are returned):
    {
        "result": [
            [
                {"document": doc, "score": score},
                ...
            ],
            ...
        ]
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk

    results, scores = retriever.batch_search(
        query_list=request.queries, num=request.topk, return_score=request.return_scores
    )

    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            combined = []
            doc_scores = zip(single_result, scores[i])
            if len(single_result) != len(scores[i]):
                warnings.warn(
                    f"Result/doc length mismatch for query {i}: docs={len(single_result)}, scores={len(scores[i])}",
                    stacklevel=2,
                )
                min_len = min(len(single_result), len(scores[i]))
                doc_scores = zip(single_result[:min_len], scores[i][:min_len])
            for doc, score in doc_scores:
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the BGE-M3 faiss retriever server.")
    parser.add_argument("--index_path", type=str, required=True, help="Path to the FAISS index file.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the local corpus JSONL file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages per query.")
    parser.add_argument("--retriever_model", type=str, default="BAAI/bge-m3", help="Path or name of the BGE-M3 model.")
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for FAISS index search.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")

    args = parser.parse_args()

    config = Config(
        retrieval_topk=args.topk,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_batch_size=512,
    )

    retriever = DenseRetriever(config)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
