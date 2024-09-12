import os
import glob
tasks = ["python2java","python2cpp","java2python","java2cpp", "cpp2java","cpp2python"]
for task in tasks:
    DirPath = f"./CodeTranslation/TransAGENT/ExperimentResult/RQ5-GeneralizationEvaluation/{task}-out"
    tobeFix = "inputOutput_runCheckTestFG_Further.jsonl"
    files = [file for file in glob.glob(DirPath+'/*') if tobeFix in file]
    for file in files:
        old_filePath = file
        print(file)
        new_filePath = file.replace(tobeFix,"ICT+SynEF+SemEF.jsonl")
        print(new_filePath)
        os.rename(old_filePath, new_filePath)
# # 原始文件名
# old_name = "old_filename.txt"
# # 新文件名
# new_name = "new_filename.txt"

# # 检查文件是否存在
# if os.path.exists(old_name):
#     # 重命名文件
#     os.rename(old_name, new_name)
#     print(f"文件已从 {old_name} 重命名为 {new_name}")
# else:
#     print(f"文件 {old_name} 不存在")
