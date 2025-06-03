
from CodeBLEU.calc_code_bleu import code_bleu
import json
from tqdm import tqdm
import os
import sys
import re
import glob

# import Levenshtein



def codebleu_by_json(dataPair, lang):
    result = []
    ID = 1
    for js in tqdm(dataPair):
        hyp = [js['trans']]
        ref = [[js['reference']]]
        score = code_bleu(hyp, ref, lang)['codebleu']
        result.append({"trans": hyp, "reference":ref, "score":float(score)})
        # result[fileName] = float(score)
        # ID += 1
    sorted_scores = sorted(result, key=lambda x: x['score'])
    total_score = sum(score_dict['score'] for score_dict in sorted_scores)
    average_score = total_score / len(sorted_scores)
    print(f"Average_codeBlue: {average_score}")

def EM(dataPair):
    result = []
    for js in tqdm(dataPair):
        hyp = js['trans']
        ref = js['reference']
        if ref == hyp:
            result.append(1)
        else: result.append(0)

    sorted_scores = sorted(result)
    total_score = sum(sorted_scores)
    average_score = total_score / len(sorted_scores)
    print(f"Average_EM: {average_score}")

if __name__ == "__main__":
    dataPair = []
    tasks = [ "java2python","java2cpp", "cpp2java","cpp2python","python2java", "python2cpp"] #"python2java", "python2cpp",
    
    strategys = ['TransCoder']
    for task in tasks:
        target_lang = task.split("2")[-1]
        if target_lang == "cpp": target_lang = "c_sharp"
        for strategy in strategys:  
            dataPair = []
            jsonFile = f"./CodeTranslation/Transcoder2/translate_results_manually/results/{task}_0829.json"
            
            with open(jsonFile, 'r', encoding='utf-8') as f:
                DataList = json.load(f)
                
            for cont in DataList:
                dataPair.append({"reference":cont['reference_code'], "trans":cont['trans_code']})
            print(f"Task: {task}; strategy: {strategy}")
            codebleu_by_json(dataPair, target_lang)





