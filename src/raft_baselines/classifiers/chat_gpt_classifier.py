from typing import Dict, Optional, List, Mapping

import numpy as np
import datasets
from sentence_transformers import util
import torch
from collections import defaultdict

from raft_baselines.classifiers.in_context_classifier import InContextClassifier
from raft_baselines.utils.gpt3_utils import (
    chat_complete,
)
from raft_baselines.utils.tokenizers import GPTTokenizer
from raft_baselines.utils.embedders import OpenAIEmbedder, SentenceTransformersEmbedder

GPT3_MAX_TOKENS = 2048


class ChatGPTClassifier(InContextClassifier):
    def __init__(
        self,
        *args,
        model: str = "gpt-3.5-turbo-1106",
        similarity_embedder_type: str = "openai", # | "sentence_transformers",
        num_responses: int = 1,
        **kwargs,
    ) -> None:
        self.num_responses = num_responses
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: str = model
        if similarity_embedder_type == "sentence_transformers":
            self.similarity_embedder = SentenceTransformersEmbedder()
        elif similarity_embedder_type == "openai":
            self.similarity_embedder = OpenAIEmbedder(max_tokens=GPT3_MAX_TOKENS)
            
        tokenizer = GPTTokenizer(model)

        super().__init__(
            *args,
            tokenizer=tokenizer,
            max_tokens=GPT3_MAX_TOKENS,
            **kwargs,
        )

    def semantically_select_training_examples(
        self, target: Mapping[str, str]
    ) -> datasets.Dataset:
        formatted_examples_without_labels = tuple(
            self.format_dict(
                {col: row[col] for col in self.input_cols if col in row},
            )
            for row in self.training_data
        )
        formatted_target = self.format_dict(target)

        # adapted from https://towardsdatascience.com/semantic-similarity-using-transformers-8f3cb5bf66d6
        target_embedding = self.similarity_embedder(tuple([formatted_target]))
        example_embeddings = self.similarity_embedder(formatted_examples_without_labels)
        
        similarity_scores = util.pytorch_cos_sim(target_embedding, example_embeddings)[
            0
        ]
        
        sorted_indices = torch.argsort(-similarity_scores.to(self.device))
        return self.training_data.select(
            list(reversed(sorted_indices[: self.num_prompt_training_examples]))
        )

    def does_token_match_class(self, token: str, clas: str) -> bool:
        clas_str = (
            f"{clas}" if not self.add_prefixes else f"{self.classes.index(clas) + 1}"
        )
        
        clas_first_token_id: int = self.tokenizer(clas_str)[0]
        token_id: int = self.tokenizer(token)[0]    

        # Compare token ids rather than the raw tokens
        # because GPT2TokenizerFast represents some special characters
        # differently from the GPT-3 API
        # (e.g. the space at the beginning of the token is " " according to the API,
        # but "Ġ" according to the tokenizer.
        # Standardizing to token ids is one easy way to smooth over that difference.
        return clas_first_token_id == token_id
    
    def format_example(
        self, example: Mapping[str, str], clas: str, max_tokens: Optional[int] = None
    ) -> (Dict[str, str], Dict[str, str]):
        clas_str = (
            clas if not self.add_prefixes else f"{self.classes.index(clas) + 1}. {clas}"
        )
        output_block = f"{self.class_col}: {clas_str}"
        output_block = (
            output_block
            if max_tokens is None
            else self.tokenizer.truncate_by_tokens(output_block, max_tokens - 2)
        )
        output_block_tokens = self.tokenizer.num_tokens(output_block)
        untruncated_text = self.format_dict(example)
        input_block = (
            untruncated_text
            if max_tokens is None
            else self.tokenizer.truncate_by_tokens(
                untruncated_text, max_tokens - output_block_tokens - 1
            )
        )
        user_message = {"role": "user", "content": input_block}
        assistant_message = {"role": "assistant", "content": output_block}
        return user_message, assistant_message
    
    def format_prompt_end(
        self, target: Mapping[str, str], max_tokens: Optional[int] = None
    ) -> (Dict[str, str], Dict[str, str]):
        output_block = f"{self.class_col}:"
        output_block_tokens = self.tokenizer.num_tokens(output_block)
        untruncated_text = self.format_dict(target)
        input_block = (
            untruncated_text
            if max_tokens is None
            else self.tokenizer.truncate_by_tokens(
                untruncated_text, max_tokens - output_block_tokens - 1
            )
        )
        user_message = {"role": "user", "content": input_block}
        assistant_message = {"role": "assistant", "content": output_block}
        return user_message, assistant_message
    
    def format_prompt(
        self,
        target: Mapping[str, str],
        example_dataset: Optional[datasets.Dataset] = None,
    ) -> List[Dict[str, str]]:
        if self.truncation_params is None:
            raise ValueError("No truncation strategy provided.")

        prompt = []
        if self.instructions != "":
            prompt.append({
                "role": "system",
                "content": self.instructions,
            })
        
        # max tokens logic may be imprecise, but it is fine for our purpose because gpt-3.5-turbo-1106 has a max token length of 16k, well 
        for row in example_dataset:
            user_message, assistant_message = self.format_example(
                {col: row[col] for col in self.input_cols if col in row},
                self.class_label_to_string(row[self.class_col]),
                max_tokens=GPT3_MAX_TOKENS,
            )
            prompt.append(user_message)
            prompt.append(assistant_message)            
            
        user_message, assistant_message = self.format_prompt_end(target, GPT3_MAX_TOKENS)
        
        prompt.append(user_message)
        prompt.append(assistant_message)
        
        return prompt

    def _get_raw_probabilities(
        self,
        prompt: List[Dict[str, str]]
    ) -> List[float]:
        
        # get token ids for the class labels
        logit_bias = {}
        for clas in self.classes:
            clas_str = (
                f"{clas}"
                if not self.add_prefixes
                else f"{self.classes.index(clas) + 1}"
            )

            clas_first_token_id: int = self.tokenizer(clas_str)[0]
            logit_bias[clas_first_token_id] = 100.0
                
        response = chat_complete(
            prompt,
            temperature=0.0,
            model=self.model,
            max_tokens=1,
            logit_bias=logit_bias,
            n=self.num_responses,
        )
                
        print(f"prompt_tokens: {response.usage.prompt_tokens}, completion_tokens: {response.usage.completion_tokens}")
        
        choice_count: Mapping[str, int] = defaultdict(int)
        num_choices = len(response.choices)
        for choice in response.choices:
            choice_count[choice.message.content] += 1
                
        raw_p = []
        for clas in self.classes:
            p = 0.0
            for choice, count in choice_count.items():
                if self.does_token_match_class(choice, clas):
                    p += count / num_choices
                
            raw_p.append(p)

        return raw_p
