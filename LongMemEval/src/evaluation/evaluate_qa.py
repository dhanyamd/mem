import os
import sys
import json
from tqdm import tqdm
import backoff
import openai
from openai import OpenAI
import numpy as np


model_zoo = {
    'llama-3.1-70b-instruct': ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'local'),
    'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 'openai'),
    'gpt-4o': ('gpt-4o-2024-08-06', 'openai'),
}


@backoff.on_exception(backoff.expo, (openai.RateLimitError,
                                    openai.APIError))
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response) 
    return prompt


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python evaluate_qa.py metric_model hyp_file ref_file')
        exit()

    metric_model_short = sys.argv[1]
    hyp_file = sys.argv[2]
    ref_file = sys.argv[3]
    verbose = True
    
    result_file = hyp_file + '.eval-results-{}'.format(metric_model_short)

    if metric_model_short not in model_zoo:
        print('Requested metric model is not supported:', metric_model_short)
        exit()
    metric_model, metric_model_source = model_zoo[metric_model_short]
    if metric_model_source == 'openai':
        openai.organization = os.getenv('OPENAI_ORGANIZATION')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        openai_api_base = None
    else:
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8001/v1"
    
    metric_client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    try:
        with open(hyp_file, 'r') as f:
            hypotheses = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Standard JSONL parse failed ({e}), trying single JSON object...")
        try:
            with open(hyp_file, 'r') as f:
                hypotheses = json.load(f)
        except:
            print("Failed to parse hypotheses file.")
            exit()

    try:
        with open(ref_file, 'r') as f:
            references = json.load(f)
    except:
        with open(ref_file, 'r') as f:
            references = [json.loads(line) for line in f if line.strip()]

    qid2qdata = {entry['question_id']: entry for entry in references}
    ref_list = references if isinstance(references, list) else list(references.values())
    
    qid2qtype = {entry['question_id']: entry.get('question_type', 'unknown') for entry in ref_list}
    qtypes = set(list(qid2qtype.values()))
    qtype2acc = {t: [] for t in qtypes}
    
    # NEW: Task-specific for Hallucination Analysis
    blur_checks = []

    def get_blur_prompt(question, answer, response):
        return f"""You are an academic evaluator. I will give you a Question, a Ground Truth Answer, and a Model Response.
Your goal is to detect 'Neural Blur' or 'Stale Data Hallucinations'. 
Does the Model Response contain information that was part of a PREVIOUS state but has been explicitly updated in the Ground Truth? 
(e.g., if Ground Truth says 'moved to London' but Model says 'lives in Melbourne and London', that is a Hallucination).

Question: {question}
Ground Truth: {answer}
Model Response: {response}

Is there a 'Stale Data' Hallucination/Blur? Answer 'Yes' or 'No' only."""

    with open(result_file, 'w') as out_f:
        logs = []
        for i, entry in enumerate(tqdm(hypotheses)):
            # Strategy: if question_id is bench_xxx_INDEX, use INDEX to find ref
            ref_item = None
            if entry['question_id'] in qid2qdata:
                ref_item = qid2qdata[entry['question_id']]
            else:
                try:
                    # Try to extract the index from the end of the question_id
                    parts = entry['question_id'].split('_')
                    idx = int(parts[-1])
                    if idx < len(ref_list):
                        ref_item = ref_list[idx]
                except (ValueError, IndexError):
                    pass
            
            if not ref_item:
                print(f"Warning: skipping {entry['question_id']} (no ref found)")
                continue
            
            qtype = ref_item.get('question_type', 'unknown')
            q = ref_item['question']
            ans = ref_item['answer']
            hyp = entry['hypothesis']
            
            # Standard Correctness
            prompt = get_anscheck_prompt(qtype, q, ans, hyp, abstention='_abs' in entry['question_id'])
            
            def call_gpt(p):
                kwargs = {
                    'model': metric_model,
                    'messages': [{"role": "user", "content": p}],
                    'temperature': 0,
                    'max_tokens': 10
                }
                completion = chat_completions_with_backoff(metric_client, **kwargs)
                return completion.choices[0].message.content.strip().lower()

            eval_response = call_gpt(prompt)
            label = 'yes' in eval_response
            
            # Blur Detection (Only for Updates/Multi-session)
            blur_label = False
            if qtype in ['knowledge-update', 'multi-session', 'unknown']:
                blur_prompt = get_blur_prompt(q, ans, hyp)
                blur_resp = call_gpt(blur_prompt)
                blur_label = 'yes' in blur_resp
                blur_checks.append(1 if blur_label else 0)

            entry['autoeval_label'] = {'model': metric_model, 'label': label, 'blur_detected': blur_label}
            logs.append(entry)
            
            print(json.dumps(entry), file=out_f)
            qtype2acc[qtype].append(1 if label else 0)

    print("\n" + "="*30)
    print("📊 RADIX-TITAN PERFORMANCE REPORT")
    print("="*30)
    if logs:
        avg_acc = np.mean([1 if x['autoeval_label']['label'] else 0 for x in logs])
        print(f'Overall Accuracy: {round(avg_acc * 100, 2)}%')
    
    if blur_checks:
        hallucination_rate = np.mean(blur_checks)
        print(f'Memory-Interference (Blur) Hallucination Rate: {round(hallucination_rate * 100, 2)}%')
    
    print("\nPer-Task Breakdown:")
    for k, v in qtype2acc.items():
        if v:
            print(f'  - {k}: {round(np.mean(v) * 100, 2)}% ({len(v)} samples)')

    print("\nSaved detailed results to:", result_file)
