import json
import os
import shutil
import datasets
from sacred import Experiment, observers
from setfit.modeling import SetFitHead
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as skm
from collections import defaultdict

from raft_baselines import classifiers

experiment_name = "loo_tuning"
raft_experiment = Experiment(experiment_name, save_git_info=False)
observer = observers.FileStorageObserver(f"results/{experiment_name}")
raft_experiment.observers.append(observer)


@raft_experiment.config
def base_config():
    classifier_name = "SetFitClassifier"
    classifier_kwargs = {
        "model_type": "sentence-transformers/paraphrase-mpnet-base-v2",
    }
    configs = datasets.get_dataset_config_names("ought/raft")
    configs.remove("ade_corpus_v2")
    configs.remove("banking_77")
    configs.remove("terms_of_service")
    
    # controls which dimension is tested, out of the 3 reported in the paper
    # Other options: do_semantic_selection and num_prompt_training_examples
    test_dimension = "model_head"
    random_seed = 42


@raft_experiment.capture
def load_datasets_train(configs):
    train_datasets = {
        config: datasets.load_dataset("ought/raft", config, split="train")
        for config in configs
    }
    return train_datasets


@raft_experiment.capture
def loo_test(
    train_datasets, classifier_name, classifier_kwargs, test_dimension, random_seed
):
    # Change what to iterate over, filling in extra_kwargs to test different
    # configurations of the classifier.

    if test_dimension == "use_task_specific_instructions":
        dim_values = [False, True]
        other_dim_kwargs = {
            "do_semantic_selection": False,
            "num_prompt_training_examples": 20,
        }
    elif test_dimension == "do_semantic_selection":
        dim_values = [False, True]
        other_dim_kwargs = {
            "use_task_specific_instructions": True,
            "num_prompt_training_examples": 20,
        }
    elif test_dimension == "num_prompt_training_examples":
        dim_values = [5, 10, 25, 49]
        other_dim_kwargs = {
            "use_task_specific_instructions": True,
            "do_semantic_selection": True,
            
        }
    elif test_dimension == "similarity_embedder_type":
        dim_values = ["sentence_transformers", "openai"]
        other_dim_kwargs = {
            "use_task_specific_instructions": True,
            "do_semantic_selection": True,
            "num_prompt_training_examples": 20,
        }
    elif test_dimension == "model_head":
        dim_values = [RandomForestClassifier, LogisticRegression]
        other_dim_kwargs = {}
    elif test_dimension == "model_type":
        dim_values = ["sentence-transformers/all-roberta-large-v1", "sentence-transformers/paraphrase-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2"]
        other_dim_kwargs = {}
    else:
        raise ValueError(f"test_dimension {test_dimension} not recognized")

    classifier_cls = getattr(classifiers, classifier_name)

    for config in train_datasets:
        for dim_value in dim_values:
            dataset = train_datasets[config]
            labels = list(range(1, dataset.features["Label"].num_classes))
            predictions = []
            test_output = {}
                        
            extra_kwargs = {
                "config": config,
                test_dimension: dim_value,
                **other_dim_kwargs,
            }
            if config == "banking_77":
                extra_kwargs["add_prefixes"] = True

            for i in range(len(dataset)):
                train = dataset.select([j for j in range(len(dataset)) if j != i])
                test = dataset.select([i])

                classifier = classifier_cls(train, **classifier_kwargs, **extra_kwargs)

                def predict(example):
                    id = example["ID"]
                    label = example["Label"]
                    del example["Label"]
                    del example["ID"]
                    output_probs = classifier.classify(example, random_seed=random_seed)
                    output = max(output_probs.items(), key=lambda kv_pair: kv_pair[1])
                    
                    test_output[id] = make_output_entry(label, output_probs, output)
                    
                    predictions.append(dataset.features["Label"].str2int(output[0]))

                test.map(predict)

            # accuracy = sum([p == l for p, l in zip(predictions, dataset['Label'])]) / 50
            f1 = skm.f1_score(
                dataset["Label"], predictions, labels=labels, average="macro"
            )
            print(f"Dataset - {config}; {test_dimension} - {dim_value}: {f1}")
            raft_experiment.log_scalar(f"{config}.{dim_value}", f1)

            save_output(test_output, config, dim_value)

def make_output_entry(label, output_probs, output):
    return {
        "label": label,
        "output_probs": output_probs,
        "output": output,
    }

def save_output(test_output, config, dim_value):
    # get file from observer
    dir = os.path.join(observer.dir, "test_outputs", str(dim_value))
    if not os.path.isdir(dir):
        os.makedirs(dir)
    
    with open(os.path.join(dir, f"{config}.json"), "w") as f:
        json.dump(test_output, f)

@raft_experiment.automain
def main():
    train = load_datasets_train()
    loo_test(train)
