####### AlignTans #####
import json
import os
from collections import Counter

def fileRead(filePath):
    DataList = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f:
            cont = json.loads(line.strip())
            DataList.append(cont)
    return DataList

tasks = ["java2python","java2cpp", "cpp2java","cpp2python","python2cpp", "python2java"] 
for task in tasks:
    print(f"Deal: {task}")
    strategys = ['inputOutput','inputOutput_compilationCheck_our', 'inputOutput_runCheckTestFG_Refine','inputOutput_runCheckTestFG_Further']
    TotalNumber = 0
    DataList_inputoutput = []
    DataList_compilation = []
    DataList_Refine = []
    DataList_Further = []
    iter1 = 0
    for strategy in strategys:
        filePath = f"./CodeTranslation/CodeTranslation/manually/{task}-out/deepseek_7B_{strategy}.jsonl"
        if strategy == "inputOutput": DataList_inputoutput = fileRead(filePath)
        elif strategy == "inputOutput_compilationCheck_our": DataList_compilation = fileRead(filePath)
        elif strategy == "inputOutput_runCheckTestFG_Refine": DataList_Refine = fileRead(filePath)
        elif strategy == "inputOutput_runCheckTestFG_Further": DataList_Further = fileRead(filePath)
    TotalNumber =  len(DataList_inputoutput)
    allData = []
    for data in DataList_Further:
        # if "java###findReplaceString_Test.java" not in data['source_Lan']: continue
        iteCount = 0
        if data['testResult'] == 1:
            source_Lan = data['source_Lan']
            inputOutput = [data for data in DataList_inputoutput if data['source_Lan'] == source_Lan and data['testResult'] == 1]
            if len(inputOutput): 
                data['iteCount'] = 0
                allData.append(iteCount)
                # allData.append({data['source_Lan']:iteCount})
            else:
                inputOutput = [data for data in DataList_compilation if data['source_Lan'] == source_Lan and data['testResult'] == 1]
                if len(inputOutput): 
                    if "NULL" in data['iterativeCount']: iteCount = 1
                    else: iteCount = data['iterativeCount']
                    data['iteCount'] = iteCount
                    allData.append(iteCount)
                    # allData.append({data['source_Lan']:iteCount})
                    
                
                else:
                    inputOutput = [data for data in DataList_compilation if data['source_Lan'] == source_Lan]
                    inputOutput_2 = [data for data in DataList_Refine if data['source_Lan'] == source_Lan and data['testResult'] == 1]
                    if len(inputOutput_2): 
                        if "iterativeCount" in inputOutput[0]: iteCount = int(inputOutput[0]['iterativeCount'])
                        iteCount = iteCount + int(inputOutput_2[0]['iterativeCount'])
                        data['iteCount'] = iteCount
                        allData.append(iteCount)
                        # allData.append({data['source_Lan']:iteCount})
                        
                    else:
                        inputOutput = [data for data in DataList_compilation if data['source_Lan'] == source_Lan]
                        inputOutput_2 = [data for data in DataList_Refine if data['source_Lan'] == source_Lan]
                        inputOutput_3 = [data for data in DataList_Further if data['source_Lan'] == source_Lan and data['testResult'] == 1]
                        if len(inputOutput_3): # 说明在最后的阶段修复正确
                            if "iterativeCount" in inputOutput[0]: iteCount = int(inputOutput[0]['iterativeCount'])
                            if "iterativeCount" in inputOutput_2[0]: iteCount = iteCount + 1
                            iteCount = iteCount + int(inputOutput_3[0]['iterativeCount'])
                            data['iteCount'] = iteCount
                            allData.append(iteCount)
                            # allData.append({data['source_Lan']:iteCount})
                            
    sorted_list = sorted(allData)
    element_counts = Counter(sorted_list)
    for key, value in element_counts.items():
        print(f"key: {key}, count: {value/TotalNumber:.3f}") 