# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import ast
import builtins
import math
import os
import tempfile
from collections import Counter
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import ray
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from nemo_rl.tools.interfaces import ToolInterface


@ray.remote
class StatefulCodeExecutor(ToolInterface):
    """Stateful code executor.

    Args:
        context: classes, functions and variables accessible to the code executor.
            By passing tools in context, the code executor also serves tool use.
    """

    def __init__(self, context: Dict[str, Any] = {}):
        self.context = context.copy()
        self.tmp_dir = tempfile.TemporaryDirectory()

        builtin_dict = {k: getattr(builtins, k) for k in dir(builtins)}
        builtin_dict["open"] = self.safe_open
        builtin_dict["__import__"] = self.safe_import
        self.sandbox = {"__builtins__": builtin_dict}

    def __call__(self, code: str) -> Optional[str]:
        tree = ast.parse(code)

        if tree.body and isinstance(tree.body[-1], ast.Expr):
            # interactive mode
            code = ast.unparse(tree.body[:-1])
            expr = ast.unparse(tree.body[-1])
        else:
            # silent mode
            expr = None

        try:
            # isolate the code in a sandbox with globals={}
            # capture local variables in self.context
            with self.change_temporary_dir():
                exec(code, self.sandbox, self.context)
                if expr:
                    return eval(expr, self.sandbox, self.context)
        except Exception as err:
            return err

    @contextmanager
    def change_temporary_dir(self):
        current_dir = os.getcwd()
        os.chdir(self.tmp_dir.name)
        try:
            yield
        finally:
            os.chdir(current_dir)

    def safe_open(self, file, *args, **kwargs):
        real_file = os.path.realpath(file)
        tmp_dir = os.path.realpath(self.tmp_dir.name)
        if os.path.commonpath([real_file, tmp_dir]) != tmp_dir:
            # real_file is not inside tmp_dir
            raise PermissionError(
                "Access beyond the temporary working directory is blocked"
            )
        return open(file, *args, **kwargs)

    def safe_import(self, name, *args, **kwargs):
        risky_modules = {
            "os",
            "shutil",  # erase filesystem
            "sys",
            "signal",  # exit the current program
            "socket",  # network communication
            "subprocess",
            "threading",
            "multiprocessing",  # spawn threads or processes
            "builtins",
            "importlib",  # bypass current blockers
        }
        if name in risky_modules:
            raise PermissionError("Importing system and network modules is blocked")
        return builtins.__import__(name, *args, **kwargs)


class BM25Retriever(ToolInterface):
    """Sparse BM25 retriever.

    Args:
        documents: list of documents to retrieve from
        num_result: retrieve top-k documents
        k1: parameter of BM25. Values in [1.2, 2.0] are recommended.
        b: parameter of BM25. 0.75 is recommended.
        device: device to compute BM25
    """

    def __init__(
        self,
        documents: List[str] = None,
        num_result: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
        device: str = "cpu",
    ):
        if documents is None:
            dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
            self.documents = [sample["text"] for sample in dataset["train"]]
        else:
            self.documents = documents
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", use_fast=True
        )
        self.num_result = num_result
        self.k1 = k1
        self.b = b
        self.device = device
        self.corpus_size = len(self.documents)
        self.vocab_size = self.tokenizer.vocab_size

        self.build_index()

    def build_index(self):
        doc_ids = []
        token_ids = []
        tfs = []
        lengths = []

        for i, document in enumerate(
            tqdm(self.documents, "Build index for BM25Retriever")
        ):
            input_ids = self.tokenizer.encode(document, add_special_tokens=False)
            token2cnt = Counter(input_ids)
            token_ids += token2cnt.keys()
            tfs += token2cnt.values()
            doc_ids += [i] * len(token2cnt)
            lengths.append(len(input_ids))

        avg_dl = sum(lengths) / self.corpus_size
        for i, doc_id in enumerate(doc_ids):
            tfs[i] = (
                tfs[i]
                * (self.k1 + 1)
                / (tfs[i] + self.k1 * (1 - self.b + self.b * lengths[doc_id] / avg_dl))
            )

        indices = torch.tensor([doc_ids, token_ids], device=self.device)
        values = torch.tensor(tfs, device=self.device)
        self.doc_tfs = torch.sparse_coo_tensor(
            indices, values, (self.corpus_size, self.vocab_size)
        )

        idfs = [0] * self.vocab_size
        token2df = Counter(token_ids)
        for token_id, df in token2df.items():
            idfs[token_id] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
        self.idfs = idfs

    def __call__(self, query: str) -> List[str]:
        input_ids = self.tokenizer.encode(query, add_special_tokens=False)
        token2cnt = Counter(input_ids)
        token_ids = []
        query_idfs = []
        for token_id, query_tf in token2cnt.items():
            token_ids.append(token_id)
            query_idfs.append(query_tf * self.idfs[token_id])

        indices = torch.tensor([token_ids, [0] * len(token_ids)], device=self.device)
        values = torch.tensor(query_idfs, device=self.device)
        query_idfs = torch.sparse_coo_tensor(indices, values, (self.vocab_size, 1))

        scores = torch.sparse.mm(self.doc_tfs, query_idfs)
        scores = scores.to_dense().squeeze(-1)
        results = []
        for i in scores.topk(k=self.num_result).indices.tolist():
            results.append(self.documents[i])

        return results
