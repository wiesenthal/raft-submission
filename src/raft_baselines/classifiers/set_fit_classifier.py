from abc import abstractmethod
import random
from typing import Dict, Optional, List, Tuple, Mapping, Any, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import json
import importlib.resources
from sentence_transformers import losses
from pathlib import Path

import numpy as np
import datasets
import torch

from raft_baselines.classifiers.classifier import Classifier
from raft_baselines import data
from setfit import SetFitModel, SetFitHead, SetFitTrainer

text_data = importlib.resources.read_text(
    data, "prompt_construction_settings.jsonl"
).split("\n")
FIELD_ORDERING = json.loads(text_data[0])
INSTRUCTIONS = json.loads(text_data[1])

PRE_SAVED_MODELS = ["banking_77", "systematic_review_inclusion"]

class SetFitClassifier(Classifier):
    separator: str = "\n\n"

    def __init__(
        self,
        training_data: datasets.Dataset,
        config: str = None,
        model_type: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        model_head: Optional[Union[SetFitHead, LogisticRegression, RandomForestClassifier]] = RandomForestClassifier,
        # Training params
        loss_class: Optional[torch.nn.Module] = losses.CosineSimilarityLoss,
        batch_size: int = 16,
        num_epochs: int = 1,
        num_iterations: int = 20,
        max_tokens: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(training_data)
        if model_head == SetFitHead:
            use_differentiable_head = True
            non_differentiable_model_head = None
        elif model_head in (LogisticRegression, RandomForestClassifier):
            use_differentiable_head = False
            non_differentiable_model_head = model_head 
        else:
            raise ValueError("model_head must be a SetFitHead or a sklearn classifier")
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if config in PRE_SAVED_MODELS and Path(__file__).parent.parent.joinpath("models", config).exists():
            self.model = SetFitModel.from_pretrained(
                Path(__file__).parent.parent.joinpath("models", config), 
                use_differentiable_head=use_differentiable_head, 
                non_differentiable_model_head=non_differentiable_model_head) \
            .to(self.device)
        else:  
            self.model = SetFitModel.from_pretrained(
                model_type, 
                use_differentiable_head=use_differentiable_head, 
                non_differentiable_model_head=non_differentiable_model_head) \
            .to(self.device)
            
            self.model.model_body.max_seq_length = max_tokens
            
            self.trainer = SetFitTrainer(
                model=self.model,
                train_dataset=self.format_training_data(training_data),
                loss_class=loss_class,
                batch_size=batch_size,
                num_epochs=num_epochs,
                num_iterations=num_iterations,
            )
            
            self.trainer.train()
        
        if config:
            self.config: str = config
            self.input_cols: List[str] = FIELD_ORDERING[config]

    @classmethod
    def format_dict(cls, example: Mapping[str, str]) -> str:
        return "\n".join(
            [f"{k}: {v}" for k, v in example.items() if len(str(v).strip())]
        )

    def format_example(self, example: Mapping[str, str]) -> str:
        return { 
            "label": example["Label"],
            "text": self.format_dict({
                col: example[col]
                for col in self.input_cols
                if col in example
                })
        }
    
    def format_prompt(self, example: Mapping[str, str]) -> str:
        return self.format_dict({
            col: example[col]
            for col in self.input_cols
            if col in example
        })
        

    def format_training_data(self, training_data: datasets.Dataset) -> datasets.Dataset:
        return training_data.map(self.format_example)

    def _get_raw_probabilities(
        self,
        prompt: str,
    ) -> List[float]:
        return self.model.predict_proba([prompt])[0].tolist()

    def _classify_prompt(
        self,
        prompt: str,
    ) -> Dict[str, float]:
        raw_p = self._get_raw_probabilities(prompt)
        sum_p = np.sum(raw_p)
        if sum_p > 0:
            normalized_p = np.array(raw_p) / np.sum(raw_p)
        else:
            normalized_p = np.full(len(self.classes), 1 / len(self.classes))
        class_probs = {}
        for i, clas in enumerate(self.classes):
            class_probs[clas] = normalized_p[i]
        return class_probs

    def classify(
        self,
        target: Mapping[str, str],
        random_seed: Optional[int] = None,
        should_print_prompt: bool = False,
    ) -> Dict[str, float]:
        prompt = self.format_prompt(target)
        if should_print_prompt:
            print(prompt)

        return self._classify_prompt(prompt)
