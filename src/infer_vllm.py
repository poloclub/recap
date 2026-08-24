"""
This file is intended for the following two use cases:
    1. Use for batch inference
    2. Use for guardrail evaluation
        a. needs to provide the generation jsonl file
        b. (optional) save the binary reward from guardrail for filtering purposes
    3. Use for overrefusal judge
"""

# Batched inference entry point using vLLM (Ray-managed vLLM engines) for
# generation throughput.
# Config: Hydra config `config/infer.yaml` (config_name="infer"), typically
# composed with `infer/default` or `infer/guardrail`.
# Invocation: `python -m src.infer_vllm ++name=<run_name> ...` (see the
# `experiment/infer/%/.done_infer*` targets in `makefiles/infer.mk`).

import logging
import os
import re
import time
from datetime import timedelta
from pathlib import Path
from pprint import pformat

import hydra
import jsonlines
import ray
import torch
import torch.distributed as dist
import transformers
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from vllm import SamplingParams

from src.data import HydraConfigDataset, blend_datasets, can_print
from src.metric import compute_metric, guardrail_answer_to_value
from src.model import (
    LLMRayActor,
    compute_tp_size_from_model_size,
    create_vllm_engines,
    get_tokenizer,
)

from .misc import get_model_path, set_seed

logger = logging.getLogger(__name__)
transformers.utils.logging.set_verbosity_error()


@torch.no_grad()
@hydra.main(config_path="../config", config_name="infer", version_base="1.3")
def main(cfg: DictConfig):
    """Run batched vLLM inference using the resolved Hydra config ``cfg``.

    Initializes Ray and a pool of vLLM engines sized to the available GPUs,
    tokenizes the configured (or, for guardrail evaluation, a previously
    generated) dataset, dispatches generation requests across the engines,
    and writes prompts/labels/generations to a ``generation_*.jsonl`` (or
    ``guardrail_*.jsonl``) file. Optionally splits chain-of-thought from the
    final answer, saves a dataset annotated with guardrail binary rewards for
    downstream filtering, and computes evaluation metrics.
    """
    output_dir = Path(os.getcwd())  # output to experiment/infer/${name}
    set_seed(cfg.seed)
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    start_time = time.time()

    # ==== init ray ====
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}
            }
        )
    logger.info(pformat(ray.cluster_resources()))

    # ==== create vllm engines ====
    vllm_tensor_parallel_size = compute_tp_size_from_model_size()
    vllm_num_engines = int(ray.cluster_resources()["GPU"]) // vllm_tensor_parallel_size
    logger.info(
        f"Spawning a total of {vllm_num_engines} llms, each using {vllm_tensor_parallel_size} GPUs."
    )

    # ray will hijack the hydra working directory, and does not recognize pathlib directory
    model_path = get_model_path(output_dir=output_dir, model_name=cfg.model.name)

    # vllm engines
    llms = create_vllm_engines(
        num_engines=vllm_num_engines,
        tensor_parallel_size=vllm_tensor_parallel_size,
        pretrain=model_path,
        seed=cfg.seed,
        full_determinism=cfg.deepspeed.full_determinism,
        enable_prefix_caching=False,
        enforce_eager=False,
        max_model_len=cfg.infer.prompt_max_len + cfg.infer.generate_max_len,
        shared_pg=None,
        gpu_memory_utilization=cfg.infer.gpu_memory_utilization,
        vllm_enable_sleep=False,
        llm_actor_cls=LLMRayActor,
    )

    # ==== load tokenizer for post-processing
    tokenizer = get_tokenizer(
        pretrain=model_path,
        model=None,
        padding_side="left",
        use_fast=not cfg.model.disable_fast_tokenizer,
    )
    # overwrite incorrect deepseek model chat template
    if cfg.model.chat_template is not None:
        logger.info("Overwritting default chat template ...")
        tokenizer.chat_template = cfg.model.chat_template

    # ==== vllm generation params
    g_args = cfg.model.generation_args
    sampling_params = SamplingParams(
        n=cfg.infer.pass_k,
        max_tokens=cfg.infer.generate_max_len,
        top_p=g_args.top_p,
        top_k=g_args.top_k,
        temperature=g_args.temperature,
        repetition_penalty=g_args.repetition_penalty,
        skip_special_tokens=True,
        include_stop_str_in_output=True,
    )

    # ==== load inference data, guardrail dataset is purely from the conversation saved in the generation.jsonl
    for_guardrail = cfg.infer.for_guardrail
    if for_guardrail and cfg.infer.generation_filename is not None:
        generation_file = (
            f"{output_dir}/generation_{cfg.infer.generation_filename}.jsonl"
        )
        logger.info(f"Loading generation file from {generation_file} ...")
        generation_data = load_dataset(
            path="json",
            data_files={"eval": generation_file},
            split="eval",
        )

        # keep only the conversation/generation columns and rename them to the
        # prompt/response naming expected by HydraConfigDataset for guardrail scoring
        ds_col_name = ["conversation", "generation"]
        col_name_mapping = {ds_col_name[0]: "prompt", ds_col_name[1]: "response"}
        infer_data = generation_data.remove_columns(
            [col for col in generation_data.column_names if col not in ds_col_name]
        ).rename_columns(col_name_mapping)
        if can_print():
            logger.info(f"Guardrail judging dataset:\n{infer_data}")
    else:
        infer_data, _ = blend_datasets(datasets_cfg=cfg.data, seed=cfg.seed)

    # make dataset
    infer_dataset = HydraConfigDataset(
        data=infer_data,
        data_cfg=cfg.data,
        model_cfg=cfg.model,
        tokenizer=tokenizer,
        max_length=cfg.infer.prompt_max_len,
        multiturn=False,
        pretrain_mode=False,
        for_guardrail=for_guardrail,
        use_response=cfg.data.use_response,
    )

    # ==== convert all tokenized input into vllm input format
    vllm_tokenized_inputs = []
    input_prompt_label_generation = []
    for sample in infer_dataset:
        # prepare vllm LLM inputs
        vllm_tokenized_inputs.append(sample[1]["input_ids"])

        # prepare output jsonl
        input_text = tokenizer.decode(sample[1]["input_ids"], skip_special_tokens=False)
        conversation = sample[2]["conversation"]
        label = None
        if cfg.data.apply_chat_template:
            # only extract the label when we don't use_response
            if conversation[-1]["role"] == "assistant" and not cfg.data.use_response:
                label = conversation[-1]["content"]
        else:
            if len(conversation) > 1:
                label = conversation[-1]

        final_sample = {
            "input": input_text,  # input is the prompt with chat template, the actual input fed into the llm
            "label": label,
            "conversation": conversation,
        }
        input_prompt_label_generation.append(final_sample)

    # model generation, distribute requests to engines and collect responses
    refs = []
    batch_size = (len(vllm_tokenized_inputs) + len(llms) - 1) // len(llms)
    for i, llm in enumerate(llms):
        prompt_token_ids = vllm_tokenized_inputs[i * batch_size : (i + 1) * batch_size]
        refs.append(
            llm.add_requests.remote(
                sampling_params=sampling_params, prompt_token_ids=prompt_token_ids
            )
        )
    ray.get(refs)

    # Retrieve and combine results from all outputs
    output_refs = []
    for i, llm in enumerate(llms):
        output_refs.append(llm.get_responses.remote())
    outputs = [item for sublist in ray.get(output_refs) for item in sublist]

    # ==== append generation to jsonl
    # for reasoning models, we split the cot and output
    cot_completed = 0
    for item, o in zip(input_prompt_label_generation, outputs):
        generations = o.outputs  # list of CompletionOutput

        if len(generations) > 1:
            gen_texts = []
            cot_texts = []
            for g in generations:
                raw_generation = g.text
                if (
                    cfg.model.cot_end_token
                    and cfg.model.cot_end_token in raw_generation
                ):
                    cot, generation = raw_generation.split(cfg.model.cot_end_token, 1)
                    cot_texts.append(cot)
                    gen_texts.append(generation)
                    cot_completed += 1
                else:
                    gen_texts.append(raw_generation)

            item["generation"] = gen_texts
            if cot_texts:
                item["cot"] = cot_texts
        else:
            raw_generation = generations[0].text
            if cfg.model.cot_end_token and cfg.model.cot_end_token in raw_generation:
                cot, generation = raw_generation.split(cfg.model.cot_end_token, 1)
                item["cot"] = cot
                item["generation"] = generation
                cot_completed += 1
            else:
                item["generation"] = raw_generation

    if cfg.model.cot_end_token:
        logger.warning(
            f"{cot_completed}/{sum(len(o.outputs) for o in outputs)} generations have completed cot"
        )

    assert len(input_prompt_label_generation) == len(infer_dataset), (
        f"Size mismatch between saved output ({len(input_prompt_label_generation)}) and dataset ({len(infer_dataset)})"
    )

    # write to file
    if for_guardrail:
        final_path = output_dir / f"guardrail_{cfg.name}.jsonl"
    else:
        final_path = output_dir / f"generation_{cfg.name}.jsonl"
    with jsonlines.open(final_path, "w") as f:
        f.write_all(input_prompt_label_generation)

    model_name = cfg.model.name.split(
        "/"
    )[
        -1
    ].lower()  # model_name is used for selecting guardrail models, assuming guardrail models are all from HF

    # save dataset with binary rewards (can be a mixture of multiple datasets)
    if cfg.infer.save_dataset:
        new_ds_folder = Path(cfg.data_dir) / "local"
        new_ds_folder.mkdir(parents=True, exist_ok=True)
        dataset_id = f"{cfg.name}.jsonl"

        # add value col
        value = guardrail_answer_to_value(
            input_prompt_label_generation,
            label_name="label",
            prediction_name="generation",
            model_name=model_name,
        )
        # dummy position for RS
        position = [[-1] for _ in range(len(value))]
        new_ds = infer_data.add_column("value", value)
        new_ds = new_ds.add_column("position", position)

        # save dataset as jsonlines
        new_ds.to_json(new_ds_folder / dataset_id, lines=True)

        logger.info(
            f"Saving a new dataset {dataset_id} with {len(new_ds)} lines to {new_ds_folder}"
        )

    # ==== compute metric (using openai llm-as-a-judge is in a separate file)
    if cfg.infer.compute_metric:
        # Get the first key in cfg.data that matches the pattern d followed by a digit
        for key in cfg.data:
            if re.match(r"d\d+", key):
                dataset_id = cfg.data[key].id
                break
        else:
            raise ValueError(
                "No dataset key starting with 'd' and a digit found in cfg.data"
            )
        results = compute_metric(
            dataset_id,
            input_prompt_label_generation,
            label_name="label",
            prediction_name="generation",
            model_name=model_name,
        )

        logger.info(
            f"Model: {cfg.model.name}\nDataset: {dataset_id}\nEvaluation results (pass@{cfg.infer.pass_k}): {results[0]}%. Additional results: {results[1:]}"
        )

    end_time = time.time()
    duration = end_time - start_time
    time_str = str(timedelta(seconds=duration)).split(".")[0]
    logger.info(f"Inference completed in {time_str} (HH:MM:SS)")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
