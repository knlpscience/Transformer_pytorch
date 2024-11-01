from rouge_score import rouge_scorer
import numpy as np

def calculate_rouge(predictions, targets):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_1_scores, rouge_2_scores, rouge_l_scores = [], [], []
    
    for pred, tgt in zip(predictions, targets):
        scores = scorer.score(tgt, pred)
        rouge_1_scores.append(scores['rouge1'].fmeasure)
        rouge_2_scores.append(scores['rouge2'].fmeasure)
        rouge_l_scores.append(scores['rougeL'].fmeasure)
    
    avg_rouge_1 = np.mean(rouge_1_scores)
    avg_rouge_2 = np.mean(rouge_2_scores)
    avg_rouge_l = np.mean(rouge_l_scores)
    
    return avg_rouge_1, avg_rouge_2, avg_rouge_l