"""MathVista eval - transformers version (reliable, single GPU)."""
import os, re, json, argparse
import pandas as pd, numpy as np
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration as Qwen2VL, AutoProcessor
from qwen_vl_utils import process_vision_info

def extract_answer(response, question_type, answer_type):
    boxed = re.findall(r'\\boxed\{([^}]*)\}', response)
    if boxed: return boxed[-1].strip()
    if question_type == 'multi_choice':
        for p in [r'[Tt]he\s+answer\s+is\s*[:\s]*\(?([A-D])\)?',
                  r'[Aa]nswer\s*[:\s]*\(?([A-D])\)?',
                  r'\(?([A-D])\)?\s*[\.。]?\s*$']:
            m = re.search(p, response)
            if m: return m.group(1)
        letters = re.findall(r'\b([A-D])\b', response)
        if letters: return letters[-1]
    if answer_type in ('integer','float'):
        numbers = re.findall(r'[-+]?\d*\.?\d+', response)
        if numbers: return numbers[-1]
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ''

def check_answer(pred, gt, question_type, answer_type, choices=None, precision=1.0):
    pred, gt = pred.strip(), gt.strip()
    if question_type == 'multi_choice' and choices:
        letter_map = {chr(65+i): str(c).strip() for i,c in enumerate(choices)}
        pred_text = letter_map.get(pred.upper(), pred)
        if pred_text == gt or pred.upper() == gt.upper(): return True
        for letter, text in letter_map.items():
            if text == gt and pred.upper() == letter: return True
        return False
    if answer_type == 'integer':
        try: return int(float(pred)) == int(float(gt))
        except: return pred == gt
    if answer_type == 'float':
        try:
            dec = int(precision) if precision >= 1 else max(0, round(-np.log10(precision)))
            return round(float(pred), dec) == round(float(gt), dec)
        except: return pred == gt
    return pred.lower() == gt.lower()

def run_inference(args):
    df = pd.read_parquet(args.data_path)
    if args.num_samples: df = df.head(args.num_samples)
    print(f'Total: {len(df)}')
    done = set()
    if os.path.exists(args.output_file):
        with open(args.output_file) as f:
            for line in f: done.add(json.loads(line)['pid'])
        print(f'Resuming: {len(done)} done')
    model = Qwen2VL.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map='auto')
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'a') as fout:
        for i in tqdm(range(len(df)), desc='Inference'):
            row = df.iloc[i]
            pid = str(row['pid'])
            if pid in done: continue
            image = Image.open(BytesIO(row['decoded_image']['bytes'])).convert('RGB')
            messages = [{'role':'user','content':[
                {'type':'image','image':image},
                {'type':'text','text':row['query']}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img_inputs, vid_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=img_inputs, videos=vid_inputs,
                padding=True, return_tensors='pt').to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=2048, temperature=0.01,
                    top_p=0.001, do_sample=True)
            response = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            result = {
                'pid': pid, 'query': row['query'], 'response': response,
                'answer': str(row['answer']), 'question_type': row['question_type'],
                'answer_type': row['answer_type'],
                'choices': row['choices'].tolist() if row['choices'] is not None else None,
                'precision': float(row['precision']) if not pd.isna(row['precision']) else 1.0,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            fout.flush()
    print(f'Done: {args.output_file}')

def run_evaluation(args):
    results = []
    with open(args.input_file) as f:
        for line in f: results.append(json.loads(line))
    print(f'Evaluating {len(results)}...')
    correct, total, by_type, details = 0, 0, {}, []
    for r in results:
        pred = extract_answer(r['response'], r['question_type'], r['answer_type'])
        gt = r['answer']
        hit = check_answer(pred, gt, r['question_type'], r['answer_type'],
                           r.get('choices'), r.get('precision', 1.0))
        correct += int(hit); total += 1
        qt = r['question_type']
        by_type.setdefault(qt, {'correct':0,'total':0})
        by_type[qt]['correct'] += int(hit); by_type[qt]['total'] += 1
        details.append({'pid':r['pid'],'pred':pred,'gt':gt,'hit':hit,
            'question_type':qt,'answer_type':r['answer_type']})
    acc = correct/total if total else 0
    print(f'\nOverall: {acc:.4f} ({correct}/{total})')
    for qt, s in by_type.items():
        print(f'  {qt}: {s[correct]/s[total]:.4f} ({s[correct]}/{s[total]})')
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump({'overall_accuracy':acc,'correct':correct,'total':total,
            'by_question_type':{k:{**v,'accuracy':v['correct']/v['total']} for k,v in by_type.items()},
            'details':details}, f, indent=2, ensure_ascii=False)
    print(f'Saved: {args.output_file}')

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='mode')
    ip = sub.add_parser('infer')
    ip.add_argument('--model-path', required=True)
    ip.add_argument('--data-path', required=True)
    ip.add_argument('--output-file', required=True)
    ip.add_argument('--num-samples', type=int, default=None)
    ep = sub.add_parser('eval')
    ep.add_argument('--input-file', required=True)
    ep.add_argument('--output-file', required=True)
    args = p.parse_args()
    if args.mode == 'infer': run_inference(args)
    elif args.mode == 'eval': run_evaluation(args)
    else: p.print_help()

if __name__ == '__main__': main()
