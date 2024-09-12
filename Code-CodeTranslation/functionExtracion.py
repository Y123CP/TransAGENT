
from Dataloader import transCoder_st, codeFuse
import os
import json

def functionExtraction(datasetName, source_lang, target_lang, strategy ="inputOutput"):
    if datasetName == "transCoder_st":
        trancoder = transCoder_st(source_lang, target_lang, strategy)
        DataList = trancoder.file_analyzer()
        outputDir = "./CodeTranslation/CodeTranslation/CodeTransInputOutput/transCoder"
        outputPath = os.path.join(outputDir, source_lang+"_data.json")
        with open(outputPath, 'w', encoding='utf-8') as f:
            json.dump(DataList, f, indent=2)
    elif datasetName == "codefuse":
        codefuse = codeFuse(source_lang, target_lang, strategy)
        DataList = codefuse.file_analyzer()
        outputDir = "./CodeTranslation/CodeTranslation/CodeTransInputOutput/codeFuse"
        outputPath = os.path.join(outputDir, source_lang+"_data.json")
        with open(outputPath, 'w', encoding='utf-8') as f:
            json.dump(DataList, f, indent=2)

dataNames = ['transCoder_st','codefuse']
langs = ['java2python','python2java','cpp2java']
for data in dataNames:
    for lang in langs:
        source_lang = lang.split("2")[0]
        target_lang = lang.split("2")[-1]
        functionExtraction(data, source_lang, target_lang)
        