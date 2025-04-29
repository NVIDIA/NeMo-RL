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
from collections import UserDict
from copy import deepcopy
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar, Union

import torch
from typing_extensions import Self

from nemo_rl.distributed.collectives import (
    gather_jagged_object_lists,
    rebalance_nd_tensor,
)

DictT = TypeVar("DictT", bound=Dict[str, Any])


class BatchedDataDict(UserDict, Generic[DictT]):
    @classmethod
    def from_batches(
        cls: Self,
        batches: List[Dict],
        pad_value_dict: Optional[Dict[str, int]] = None,
    ) -> Self:
        """Given a list of batches, stack the tensors/lists within and put them in a single dictionary.

        Pad sequences to the max length in the batch using either 0(default) or a non-default value for a given key provided in pad_value_dict.

        Args:
            batches (List[Dict]): A list of dictionaries, each containing a batch of data.
            pad_value_dict (Optional[Dict[str, int]]): An optional dict mapping keys to non-default(0) padding values.

        Returns:
            BatchedDataDict: A new BatchedDataDict containing the stacked data.
        """
        stacked_dict = cls()
        pad_value_dict = pad_value_dict or {}

        for k in sorted(batches[0]):
            list_of_tensors = [item[k] for item in batches]

            if isinstance(list_of_tensors[0], list):
                tensor = [item for sublist in list_of_tensors for item in sublist]
            elif all(x.ndim == 1 for x in list_of_tensors):
                tensor = torch.cat(list_of_tensors)
            elif isinstance(list_of_tensors[0], torch.Tensor):
                pad_value = pad_value_dict.get(k, 0)

                list_of_tensors = [
                    row.flatten() for tensor in list_of_tensors for row in tensor
                ]
                # TODO: can we avoid padding locally then padding globally?
                tensor = torch.nn.utils.rnn.pad_sequence(
                    list_of_tensors, batch_first=True, padding_value=pad_value
                )
            else:
                raise NotImplementedError(
                    (
                        f"Attempted to stack for unsupported type {type(list_of_tensors[0])} with key {k}."
                        "Please provide either a tensor or a list of picklable objects."
                    )
                )
            stacked_dict[k] = tensor

        return stacked_dict

    def all_gather(self, group: torch.distributed.ProcessGroup) -> "BatchedDataDict":
        """Gathers batches with possibly jagged leading dimensions across the DP ranks.

        If using reshard, it will treat PP as DP ranks.
        Works with data that is either tensors or string lists.
        """
        global_rollout_batch = type(self)()

        for k, value in self.data.items():
            if isinstance(value, torch.Tensor):
                value = rebalance_nd_tensor(value, group=group)
                global_rollout_batch[k] = value
            elif isinstance(value, list):
                value = gather_jagged_object_lists(value, group=group)
                global_rollout_batch[k] = value
            else:
                raise NotImplementedError(
                    (
                        f"Attempted to gather_and_balance_globally for unsupported type {type(value)} with key {k}."
                        "Please provide either a tensor or a list of picklable objects."
                    )
                )

        return global_rollout_batch

    def chunk(self, rank: int, chunks: int) -> "SlicedDataDict":
        """Chunks a global batch into 'chunks' splits and returns the 'rank'th split batch=[A A A B B B D D E], rank=2, chunks=3 -> [D D E].

        Requires all leading dimensions of tensors and lengths of lists to be the same over the batch
        and the chunks must divide batch size.
        """
        chunked_batch = SlicedDataDict()

        batch_set = set()
        for val in self.data.values():
            if isinstance(val, torch.Tensor):
                batch_set.add(val.size(0))
            else:
                batch_set.add(len(val))

        assert len(batch_set) == 1, (
            "batch sizes are not the same across the rollout batch"
        )
        B = batch_set.pop()
        assert B % chunks == 0, (
            f"batch size ({B}) is not a multiple of chunks ({chunks})"
        )
        assert B // chunks > rank, (
            f"index OOB: not enough splits for this rank. rollout_batch_size: {B}, chunks ({chunks}), rank_idx ({rank})"
        )

        indices = torch.arange(B).tensor_split(chunks)[rank]

        for k in self.data:
            if torch.is_tensor(self.data[k]):
                chunked_batch[k] = self.data[k][indices].clone()
            else:
                chunked_batch[k] = [self.data[k][i] for i in indices]

        return chunked_batch

    def shard_by_batch_size(
        self,
        shards: int,
        batch_size: Optional[int] = None,
        allow_uneven_shards: bool = False,
        batch_size: Optional[int] = None, 
        dynamic_batching_cfg = None
    ) -> List["SlicedDataDict"]:
        """Shards a batch by first dividing it into chunks of size batch_size, then further dividing each chunk into shards equal parts. Finally aggregates the sub-shards by their position.

        If batch_size is None, there will be no chunking beforehand (will default to the total batch size).

        For example, with data [A A B B C C D D], batch_size=2, shards=2:
        - Element 0: [A B C D] (first elements from each chunk)
        - Element 1: [A B C D] (second elements from each chunk)

        Args:
            shards (int): The number of shards to divide each batch_size chunk into.
            batch_size (int): The size of each initial chunk.
            allow_uneven_shards (bool): Whether to allow shards to be unevenly sized.
                                        If True, the last shard may be smaller than the others.

        Returns:
            List[BatchedDataDict]: A list of BatchedDataDicts, length equal to shards.

        Examples:
        ```{doctest}
        >>> from nemo_rl.distributed.batched_data_dict import BatchedDataDict
        >>> # Create a batch of two message logs with different lengths
        >>> batch = BatchedDataDict({
        ...     'problem_id': [0, 0, 1, 1, 2, 2, 3, 3],
        ...     'arbitrary_data': [1, 2, 3, 4, 5, 6, 7, 8]
        ... })
        >>> shards = batch.shard_by_batch_size(shards=2)
        >>> shards
        [{'problem_id': [0, 0, 1, 1], 'arbitrary_data': [1, 2, 3, 4]}, {'problem_id': [2, 2, 3, 3], 'arbitrary_data': [5, 6, 7, 8]}]
        >>> # Now say that I'm training with a GBS of 4 and I want to take gradients steps on problems 0 and 1 before 2 and 3 (problems are repeated because GRPO)
        >>> # In the current case, problems 0 and 2 will be trained on first since they're the first elements in each DP rank's batch.
        >>> # So, we'll use the batch_size argument to split the batch into chunks of size 4 first.
        >>> shards = batch.shard_by_batch_size(shards=2, batch_size=4)
        >>> shards
        [{'problem_id': [0, 0, 2, 2], 'arbitrary_data': [1, 2, 5, 6]}, {'problem_id': [1, 1, 3, 3], 'arbitrary_data': [3, 4, 7, 8]}]
        >>> # Now, the ranks have 0 and 1 first so when they split their batches into microbatches (of size 2 since GBS=4 and DP=2), they'll train on 0 and 1 first.
        >>> # Another way to use this function is with the 'allow_uneven_shards' flag, which allows the last shard to be smaller than the others when necessary.
        >>> # This is necessary in multi-turn rollouts when some sequences terminate early, leaving unclean batch sizes.
        >>> batch = BatchedDataDict({
        ...     'problem_id': [0, 1, 2, 3, 4],
        ...     'arbitrary_data': [10, 11, 12, 13, 14]
        ... })
        >>> shards = batch.shard_by_batch_size(shards=2, allow_uneven_shards=True)
        >>> shards
        [{'problem_id': [0, 1, 2], 'arbitrary_data': [10, 11, 12]}, {'problem_id': [3, 4], 'arbitrary_data': [13, 14]}]
        >>> # This is incompatible with the batch_size argument
        ```
        """
        if allow_uneven_shards:
            assert batch_size is None, (
                "batch_size must be None if allow_uneven_shards is True"
            )

        # Get the total batch size
        batch_sizes = set()
        for val in self.data.values():
            if isinstance(val, torch.Tensor):
                batch_sizes.add(val.size(0))
            else:
                batch_sizes.add(len(val))

        assert len(batch_sizes) == 1, (
            "Batch sizes are not the same across the rollout batch"
        )
        total_batch_size = batch_sizes.pop()
        if batch_size is None:
            batch_size = total_batch_size

        # Validate that our batch size parameters are compatible with the data dimensions
        assert total_batch_size % batch_size == 0, (
            f"Total batch size ({total_batch_size}) is not a multiple of batch_size ({batch_size})"
        )
        if not allow_uneven_shards:
            assert batch_size % shards == 0, (
                f"Batch size ({batch_size}) is not a multiple of shards ({shards})"
            )

        num_chunks = total_batch_size // batch_size
        # Calculate shard size, rounding up if not evenly divisible
        shard_size = (
            (batch_size + shards - 1) // shards
            if allow_uneven_shards
            else batch_size // shards
        )
        shard_size = batch_size // shards

        if dynamic_batching_cfg is not None:
            assert 'sequence_lengths_per_input' in dynamic_batching_cfg
            assert 'max_tokens_per_microbatch' in dynamic_batching_cfg
            assert len(dynamic_batching_cfg['sequence_lengths_per_input']) == total_batch_size
            
        # batch shard is not inplace, create a deepcopy of the data
        data = {}
        for k, v in self.data.items():
            if torch.is_tensor(v):
                data[k] = torch.clone(v)
            else:
                import copy
                data[k] = copy.deepcopy(v)

        # if using dynamic microbatching, preprocess the data by sorting the data 
        # by the sequence lengths. This ensures each DP rank receives samples of about 
        # equal sequence lengths which improves load balancing
        if dynamic_batching_cfg is not None:
            batch_sorted_indices = []
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * batch_size
                chunk_end = (chunk_idx + 1) * batch_size
                chunk_seqlens = dynamic_batching_cfg['sequence_lengths_per_input'][chunk_start:chunk_end]
                # sort the indices by sequence lengths
                chunk_idx_indices = sorted(range(batch_size), key=lambda i: chunk_seqlens[i])
                chunk_idx_indices = [i + chunk_start for i in chunk_idx_indices]
                # stride the sorted sequence lengths along the shards
                chunk_idx_indices = [chunk_idx_indices[i::shards] for i in range(shards)]
                chunk_idx_indices = sum(chunk_idx_indices, [])
                # append the sorted sequence lengths for the chunk
                batch_sorted_indices.extend(chunk_idx_indices)

            # finally reorder the data along the sorted sequence len indices
            for k,v  in data.items():
                if torch.is_tensor(v):
                    sorted_v = v.index_select(
                        dim=0, index=torch.IntTensor(batch_sorted_indices))
                else:
                    sorted_v = [v[i] for i in batch_sorted_indices]

                data[k] = sorted_v

        # Create one BatchedDataDict per shard position
        aggregated_shards = [SlicedDataDict() for _ in range(shards)]

        # Group data by shard position across all chunks
        for shard_idx in range(shards):
            for chunk_idx in range(num_chunks):
                # Calculate indices for this particular sub-shard within the chunk
                chunk_start = chunk_idx * batch_size
                shard_start = chunk_start + shard_idx * shard_size
                shard_end = chunk_start + (shard_idx + 1) * shard_size
                if allow_uneven_shards:
                    # Cap the end index at the total batch size for the last shard
                    # or if shard_end calculation goes beyond total_batch_size
                    shard_start = min(shard_start, total_batch_size)
                    shard_end = min(shard_end, total_batch_size)
                indices = torch.arange(shard_start, shard_end)

                for k in data:
                    if k not in aggregated_shards[shard_idx]:
                        # First time seeing this key for this shard, initialize it
                        if torch.is_tensor(data[k]):
                            aggregated_shards[shard_idx][k] = data[k][
                                indices
                            ].clone()
                        else:
                            aggregated_shards[shard_idx][k] = [
                                data[k][i] for i in indices
                            ]
                    else:
                        # Append to existing data - concatenate tensors or extend lists
                        if torch.is_tensor(data[k]):
                            aggregated_shards[shard_idx][k] = torch.cat(
                                [
                                    aggregated_shards[shard_idx][k],
                                    data[k][indices].clone(),
                                ]
                            )
                        else:
                            aggregated_shards[shard_idx][k].extend(
                                [data[k][i] for i in indices]
                            )
        
   
        # map inputs to microbatches such that the total number tokens in 
        # a microbatch is as close to (including padding tokens) 'max_tokens_per_microbatch'
        if dynamic_batching_cfg is not None:
            max_tokens_per_microbatch = dynamic_batching_cfg['max_tokens_per_microbatch']
            microbatch_indices = []
            # loop through each chunk, dividing the chunk into microbatches
            for chunk_idx in range(num_chunks):
                chunk_microbatch_indices = [[0,1]]
                current_mbs_total_tokens = 0
                #for each indice in the shard, map it to an microbatch
                for shard_indice in range(shard_size):
                    # use the max seqlen of all shards to calculate the total number of tokens in the mb
                    # this ensures each DP rank has the same batch size each iteration which is 
                    # required for FSDP2 and megatron policies.
                    chunk_start = chunk_idx*shard_size
                    chunk_end = (chunk_idx+1)*shard_size
                    seqlens_this_chunk = [
                        shard['input_lengths'][chunk_start:chunk_end][shard_indice]
                        for shard in aggregated_shards
                    ]
                    # check if the sample at shard_indice may be added to the current mbs for all shards
                    # the total tokens of a mbs = number of indices in the mbs * the max sequence length in the mbs
                    max_seqlen_this_chunk = max(seqlens_this_chunk).item()
                    curr_mbs_size = chunk_microbatch_indices[-1][1] - chunk_microbatch_indices[-1][0] + 1
                    total_tokens_in_mbs = curr_mbs_size * max_seqlen_this_chunk
                    # if the current mbs can accomodate this indice, add it
                    if total_tokens_in_mbs <= max_tokens_per_microbatch:
                        chunk_microbatch_indices[-1][-1] = shard_indice+1
                        current_mbs_total_tokens += max_seqlen_this_chunk
                    # otherwise start a new mbs
                    else:
                        chunk_microbatch_indices.append([shard_indice, shard_indice+1])
                        current_mbs_total_tokens = max_seqlen_this_chunk

                microbatch_indices.append(chunk_microbatch_indices)
            return aggregated_shards, microbatch_indices

        return aggregated_shards    

    def slice(self, start: int, end: int) -> "SlicedDataDict":
        """Slices the batch from start to end.

        Args:
            start: Starting index (inclusive)
            end: Ending index (exclusive)

        Returns:
            BatchedDataDict: A new BatchedDataDict containing the sliced data
        """
        sliced_batch = SlicedDataDict()
        for k in self.data:
            sliced_batch[k] = self.data[k][start:end]
        return sliced_batch

    def repeat_interleave(self, num_repeats: int) -> "BatchedDataDict":
        """Repeats the batch num_repeats times.

        For each element in the batch, repeat each value num_repeats times.
        i.e:
        {"key": torch.tensor([1, 2, 3]), "other_key": [1, 2, 3]} -> {"key": torch.tensor([1, 1, 2, 2, 3, 3]), "other_key": [1, 1, 2, 2, 3, 3]}
        """
        repeated_batch = BatchedDataDict()
        for k, v in self.data.items():
            if torch.is_tensor(v):
                # For tensors, use repeat_interleave to repeat each element
                repeated_batch[k] = v.repeat_interleave(num_repeats, dim=0)
            else:
                # For lists or other sequences, use a list comprehension to repeat each element
                repeated_batch[k] = [
                    deepcopy(item) for item in v for _ in range(num_repeats)
                ]
        return repeated_batch

    def make_microbatch_iterator_from_indices(
        self, 
        microbatch_indices,
    ) -> Iterator["SlicedDataDict"]:
        for start, end in microbatch_indices:
            yield self.slice(start, end)    
        
    def make_microbatch_iterator(
        self, 
        microbatch_size: int = None, 
    ) -> Iterator["SlicedDataDict"]:
        """Make an iterator over the batch that yields microbatches."""
        bsize = self.size
        assert microbatch_size is not None
        assert bsize % microbatch_size == 0, (
            f"Data dict size ({bsize}) is not a multiple of the provided microbatch size ({microbatch_size})"
        )
        for i in range(0, bsize, microbatch_size):
            yield self.slice(i, i + microbatch_size)

    @property
    def size(self) -> int:
        """Get the batch size of the batch."""
        # Get the first key and use its size as the batch size
        # This assumes all keys have the same batch size
        key = next(iter(self.data))
        if not self.data:
            return 0
        if not torch.is_tensor(self.data[key]):
            return len(self.data[key])
        return self.data[key].shape[0]

    def to(self, device: torch.device) -> Self:
        """Move tensors in batched dict to device."""
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.to(device)
        return self

    def select_indices(
        self, indices: Union[List[int], torch.Tensor]
    ) -> "BatchedDataDict":
        """Selects specific rows from the batch based on indices.

        Args:
            indices: A list or tensor of integer indices to select.

        Returns:
            BatchedDataDict: A new BatchedDataDict containing only the selected rows.
        """
        selected_batch = BatchedDataDict()
        for k, v in self.data.items():
            if torch.is_tensor(v):
                selected_batch[k] = v[indices]
            elif isinstance(v, list):
                selected_batch[k] = [v[i] for i in indices]
            else:
                # Handle other potential types if necessary, or raise error
                raise TypeError(
                    f"Unsupported type {type(v)} for index selection in BatchedDataDict"
                )
        return selected_batch

    def get_dict(self) -> dict:
        """Get the underlying data dictionary."""
        return self.data


class SlicedDataDict(BatchedDataDict):
    """A specialized subclass of BatchedDataDict that represents a slice or shard of a larger batch.

    This class provides a distinct type to differentiate between full batches and sliced/sharded batches, which can be helpful for
    type checking.
    """

    pass
