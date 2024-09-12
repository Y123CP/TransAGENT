import json
import csv
tasks = ["java2python","java2cpp", "cpp2java","cpp2python","python2cpp", "python2java"] #"python2java", "python2cpp", 

outputPath = "./CodeTranslation/CodeTranslation/codeAlignment/Alignmented/294Data.csv"
for task in tasks:
    filePath = f"./CodeTranslation/CodeTranslation/codeAlignment/Alignmented/{task}.jsonl"
    
    source_commentTag = "#" if "python" in  task.split("2")[0] else "//"
    tag = task.split("2")[0] if task.split("2")[0] != "python" else "py"
    
    DataList = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f:
            cont = json.loads(line.strip())
            DataList.append(cont)
    DealedData = []
    with open(outputPath, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['source_code_str', 'trans_code', 'AlignTrans_alignment', 'TransMap_alignment', 'lineCode']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for data in DataList:
            source_code_str = data['source_code_str']
            trans_code = data['trans_code']
            
            source_code_list = source_code_str.split("\n")
            stmt_source_code_list = []
            count = 1
            for code in source_code_list:
                stmt = f" {source_commentTag} --- {tag} stmt {count}"
                if not code.strip(): continue
                new_code = code + stmt
                stmt_source_code_list.append(new_code)
                count = count + 1
            stmt_source_code_str = "\n".join(stmt_source_code_list)
            
            
            
            
            
            if "TransMap_alignment" not in data.keys(): continue
            TransMap_alignment = data['TransMap_alignment']
            if data['source_Lan'] in DealedData: continue
            
            AlignTrans_alignment = [data['AlignTrans_alignment'] for data in DataList if source_code_str in data['source_code_str'] and "AlignTrans_alignment" in data][0]
            DealedData.append(data['source_Lan'])

            writer.writerow({
                'source_code_str': source_code_str,
                'trans_code': trans_code,
                'AlignTrans_alignment': AlignTrans_alignment,
                'TransMap_alignment': TransMap_alignment,
                'lineCode':stmt_source_code_str
            })

