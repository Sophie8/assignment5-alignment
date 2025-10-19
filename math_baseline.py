from typing import Callable, List
from vllm import LLM, SamplingParams
import pandas as pd
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def load_data(path_to_dataset: str):
    df = pd.read_parquet(path_to_dataset)
    print("total number of data points: ", df.shape)
    print("sampled data: ", df.head())
    return df.iloc[0:200]


def zero_shot_generate(path_to_model: str, data: pd.DataFrame, prompt_format_path: str):
    prompts = []
    with open(prompt_format_path, 'r') as f:
        prompt_template = f.read()
        prompts = data['problem'].apply(lambda x: prompt_template.format(question=x)).tolist()
    
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )
    # Create an LLM.
    llm = LLM(model=path_to_model, max_num_seqs=2, gpu_memory_utilization=0.9, enforce_eager=True)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    output_to_json = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        output_to_json.append(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    with open("zero_shot_math_output.json", "w") as f:
        json.dump(output_to_json, f, indent=4)

def evaluate_vllm(
    reward_fn: Callable[[str, str], dict[str, float]],
    outputs: List[str],
    ground_truthes: List[str]
    ) -> list[list]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    total_score = []
    total_format_score = []
    total_answer_score = []
    for output, ground_truth in zip(outputs, ground_truthes):
        score = reward_fn(output, ground_truth, fast=False)
        total_score.append(score["reward"])
        total_format_score.append(score["format_reward"])
        total_answer_score.append(score["answer_reward"])
    return [total_score, total_format_score, total_answer_score]

    
def main():
    df = load_data('MATH.parquet')
    #zero_shot_generate("Qwen/Qwen2.5-Math-1.5B", df, "cs336_alignment/prompts/r1_zero.prompt")
    outputs = []
    with open("zero_shot_math_output.json", "r") as f:
        outputs = json.load(f)
    ground_truthes = df.iloc[0:200]['solution'].to_list()
    responses = [output.split("Generated text: ")[1] for output in outputs]
    #print("response: ", responses[2])
    #print("ground truth: ", ground_truthes[2])
    [total_score, total_format_score, total_answer_score] = evaluate_vllm(r1_zero_reward_fn, responses, ground_truthes)
    print("total reward: ", total_score)
    print("total format reward: ", total_format_score)
    print("total answer reward: ", total_answer_score)
    
    


if __name__ == "__main__":
    main()
    # command to cleanup vllm gpu ram: nvidia-smi, pkill -9 -ef "VLLM::EngineCore"