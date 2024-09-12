'''
返回如下
1. source code test input, main function of execute the source code
2. target code test input, and main function of execute the target code
'''
import sys
import torch
# torch.cuda.set_device(2) 
from transformers import LlamaForCausalLM,LlamaTokenizer,AutoModelForCausalLM,AutoTokenizer
import os
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from langchain.llms import OpenAI
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
import glob
import handcraftPrompt
import ast
import re
import traceback
import shutil
from Dataloader import transCoder_st, codeFuse, Manually
from Compile_and_Run_testInputs import Compile_and_Run
import BasicBlock
from definedTool import definedTool
current_dir = os.getcwd()
print("Current_dir: ",current_dir)
import subprocess
import time

class HuggingfaceModel:
    def __init__(self,model_name_or_path,model_type):
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.float16, low_cpu_mem_usage=True).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_type = model_type
    
    def generate(self,query,return_input=False,skip_special_tokens = True, do_sample=True,temperature =0.2)->str:
        if self.model_type == "deepseek":
            messages=[
                    { 'role': 'user', 'content': query}
                ]
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('cuda')
            # 32021 is the id of <|EOT|> token
            outputs = self.model.generate(inputs, max_new_tokens=2048, do_sample=do_sample,temperature =temperature, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
            
            return self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        else:
            query = self.process_prompt(query)
            # query = "This is a test"
            input_ids = self.tokenizer.encode(query,return_tensors='pt').to('cuda')
            input_length = len(query)
            tokens = self.model.generate(input_ids,max_new_tokens=1024, do_sample=False,temperature = 0)
            response = self.tokenizer.decode(tokens[0],skip_special_tokens=skip_special_tokens)
            if return_input==True:
                return response
            else:
                return response[input_length:]
    
    def process_prompt(self,query):
        if self.model_type == 'codellama':
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
            _system_prompt = '''
            You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            '''
            system_prompt = f"{B_SYS}{_system_prompt}{E_SYS}"
            prompt = system_prompt+f"{B_INST} {query.strip()} {E_INST} "
            return prompt
        elif self.model_type == 'starcoder':
            prefix_token = "<fim_prefix>"
            suffix_token = "<fim_suffix><fim_middle>"
            prompt = prefix_token + query + suffix_token
            return prompt

class ChatGenerationModel:
    def __init__(self):
        OPENAI_API_KEY = 'KEY'
        openai.api_key = OPENAI_API_KEY
        openai.api_base = "https://openkey.cloud/v1"'
        self.model_name = 'gpt-4o-mini'

    
    def generate(self,query,return_input=False,skip_special_tokens = True)->str:
        messages = [
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": query}
            ]
        response = openai.ChatCompletion.create(
            model = self.model_name,
            messages = messages,
        )
        answer = response.choices[0]['message']['content']
        return answer

class MockModel:
    def generate(self,query,return_input=False):
        return "111111111111111111111111"


class PromptDefine:    
    def TestInputGen(self, source_lang, source_code, Example):
        t_template = PromptTemplate(template=handcraftPrompt.TestInputGen,input_variables=['source_lang', 'source_code', 'Example'])
        template = t_template.format(source_lang =source_lang.capitalize(), source_code=source_code, Example = Example)
        return template   
    
    def TestInputGen_validTest(self, source_lang, source_code, Example):
        t_template = PromptTemplate(template=handcraftPrompt.TestInputGen_validTest,input_variables=['source_lang', 'source_code', 'Example'])
        template = t_template.format(source_lang =source_lang.capitalize(), source_code=source_code, Example = Example)
        return template  
     
    def MainFuncGen(self, source_lang, source_code, Test_Inputs, methodSig, Example):
        t_template = PromptTemplate(template=handcraftPrompt.MainFuncGen,input_variables=['source_lang', 'source_code','Test_Inputs', 'methodSig','Example'])
        template = t_template.format(source_lang =source_lang.capitalize(), source_code=source_code,Test_Inputs=Test_Inputs,methodSig=methodSig, Example = Example)
        return template
    def inputOutputArchive(self, lang, code, run_result, Example):
        t_template = PromptTemplate(template=handcraftPrompt.inputOutputArchive,input_variables=['lang', 'code', 'run_result', 'Example'])
        template = t_template.format(lang =lang.capitalize(), code=code,run_result=run_result, Example = Example)
        return template    
        
    
    
class TestGenMain:
    def __init__(self,source_lang,target_lang, model_name,strategy, dataset_name, ValidTest):
        self.model = ChatGenerationModel()
        
        self.debug = True 
        if model_name == 'gpt': self.debug = False 
        self.chatgptModel = ChatGenerationModel()
        self.promptSelect= PromptDefine()
        self.toolUse = definedTool()
        self.dataset_name = dataset_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.ValidTest = ValidTest
        if strategy == "TestGeneration":
            if dataset_name == "manually" and self.ValidTest== False: 
                mainually = Manually(source_lang, target_lang)
                DataList = mainually.file_analyzer() 
            elif self.ValidTest:
                sourceDataPath = os.path.join(output_path, f"{self.source_lang}2{self.target_lang}_data.jsonl") 
                DataList = []
                with open(sourceDataPath, 'r',encoding='utf-8') as f:
                    for line in f:
                        cont = json.loads(line.strip())
                        DataList.append(cont)
        
        if self.ValidTest:
            final_result_path = os.path.join(output_path.replace("manually","manuallyTest"), f"{self.source_lang}2{self.target_lang}_data.jsonl") 
        else:
            final_result_path = os.path.join(output_path, f"{self.source_lang}2{self.target_lang}_data.jsonl") 

        self.mainStart(DataList,strategy,final_result_path)
    # java2python --> java; python2java --> python; cpp2java --> java; cpp2python --> python
    def mainStart(self, DataList,strategy,final_result_path):
        if os.path.exists(final_result_path):
            with open(final_result_path, 'r', encoding='utf-8') as f:
                file_cont = f.read()
        else: file_cont = ""
        
        compileRun = Compile_and_Run(self.target_lang, self.dataset_name, "compilationCheck")
        if strategy == "TestGeneration":
            for Data in tqdm(DataList):  
                TAG = ""
                if Data["source_Lan"] in file_cont: continue

                print("Deal: ",Data['source_Lan'])
                
                if self.source_lang == "python":
                    target_function_code = Data['reference_code']
                    TestExample = self.ExampleSelection("TestGeneration", self.target_lang)
                    
                    if self.ValidTest: testGenPrompt = self.promptSelect.TestInputGen_validTest(self.target_lang, target_function_code, TestExample)
                    else: testGenPrompt = self.promptSelect.TestInputGen(self.target_lang, target_function_code, TestExample)
                    TestGen_response = self.model.generate(testGenPrompt, False)
                    TestInputs = self.TestInput_responseDeal(TestGen_response)
                    print("TestInputs:\n",TestInputs)
                
                else:
                    source_function_code = Data['source_code_str']
                    target_function_code = Data['reference_code']
                    TestExample = self.ExampleSelection("TestGeneration", self.source_lang)
                    if self.ValidTest: testGenPrompt = self.promptSelect.TestInputGen_validTest(self.target_lang, target_function_code, TestExample)
                    else: testGenPrompt = self.promptSelect.TestInputGen(self.target_lang, target_function_code, TestExample)
                    TestGen_response = self.model.generate(testGenPrompt, False)
                
                    TestInputs = self.TestInput_responseDeal(TestGen_response)
                    print("TestInputs:\n",TestInputs)
                    
    
                try:
                    if self.source_lang == "cpp":
                        target_codeShell, all_input_output_value, input_output_value = self.codeShell_IO_Gen(Data['target_Lan'].split("###")[-1], Data['reference_code'], self.target_lang, TestInputs, Data['target_import'], Data['target_allMS'], Data['target_method_signature'])
                        Data['target_code_shell'] = target_codeShell.strip()
                        Data['all_input_output_value'] = all_input_output_value
                        Data['input_output_value'] = input_output_value
                        if "runFailed" == target_codeShell: 
                            TAG = "ERROR_runFailed"
                            continue
    
                        source_codeShell = self.IO2CodeShell_Gen(Data['source_Lan'].split("###")[-1],Data['source_code_str'], Data['source_import'], Data['source_allMS'],Data['source_method_signature'], self.source_lang, all_input_output_value)
                        Data['source_code_shell'] = source_codeShell.strip()
                        print("Source_codeShell:\n", source_codeShell)
                        if "runFailed" == source_codeShell: 
                            TAG = "ERROR_runFailed"
                            continue
                        STATE,Group, blockNumber, InputOutput_function, _, _, _, _,_  = compileRun.mainStart(Data, "runRepair",validTest=True, boolStrategy=True)
                        if len(InputOutput_function) != 0: 
                            TAG = "ERROR_NOTEQUAL"
                    else:
                        
                        source_codeShell, all_input_output_value, input_output_value = self.codeShell_IO_Gen(Data['source_Lan'].split("###")[-1], Data['source_code_str'], self.source_lang, TestInputs, Data['source_import'], Data['source_allMS'], Data['source_method_signature'])
                        Data['source_code_shell'] = source_codeShell.strip()
                        Data['all_input_output_value'] = all_input_output_value
                        Data['input_output_value'] = input_output_value
                        if "runFailed" == source_codeShell: 
                            TAG = "ERROR_runFailed"
                            continue
                        
                        target_codeShell = self.IO2CodeShell_Gen(Data['target_Lan'].split("###")[-1],Data['reference_code'], Data['target_import'], Data['target_allMS'], Data['target_method_signature'], self.target_lang, all_input_output_value)
                        Data['target_code_shell'] = target_codeShell.strip()
                        if "runFailed" == target_codeShell: 
                            TAG = "ERROR_runFailed"
                            continue
                        STATE,Group, blockNumber, InputOutput_function, _, _, _, _,_  = compileRun.mainStart(Data, "runRepair",validTest=True, boolStrategy=True)
                        if len(InputOutput_function) != 0: 
                            TAG = "ERROR_NOTEQUAL"
                except Exception as e:
                    ErrorInfo = traceback.format_exc()
                    TAG = "ERROR_"+ErrorInfo
                    print(traceback.format_exc())
                finally:
                    
                    if len(TAG): Data['TAG'] = TAG
                    else: 
                        Data['TAG'] = "Success"
                        with open(final_result_path, 'a', encoding='utf-8') as f:
                            json.dump(Data,f)
                            f.write('\n')  
                    print("TAG:\n", TAG)
                    

    def modify_jsonl(self, final_result_path, data_to_remove, new_data):
        filtered_records = []
        with open(final_result_path, 'r') as file:
            for line in file:
                record = json.loads(line.strip())
                if record['source_Lan'].replace("_Test","") != data_to_remove.replace("_Test",""):
                    filtered_records.append(record)
                    
        filtered_records.append(new_data)
             
        with open(final_result_path, 'w') as file:
            for record in filtered_records:
                file.write(json.dumps(record) + '\n')
                    
    def codeShell_IO_Gen(self, fileName, function_code, lang, TestInputs, ImportInfo, All_MS, method_signature):

        MainFunc_temp, import_info = self.MainGene(function_code, lang, TestInputs, method_signature)
        CodeCont_temp, _ = self.GetCodeCont(MainFunc_temp, function_code, ImportInfo + "\n" + import_info ,fileName.split(".")[0], lang)
        print("MainFunc_temp\n", MainFunc_temp)                   
        ExecuteOutput, CodeCont = self.TestPrune(fileName, CodeCont_temp.strip(), lang)
        if len(ExecuteOutput) == 0:
            return "runFailed", "runFailed", "runFailed"

        if lang == "python": MainFunc = self.extract_Mainfunction_python(CodeCont)
        elif lang in ["java", "cpp"]: MainFunc = self.extract_Mainfunction_java_cpp(CodeCont)
        
        _, codeShell = self.GetCodeCont(MainFunc, function_code, ImportInfo.strip()+ "\n" + import_info, fileName.split(".")[0], lang)
        print("codeShell_IO_Gen_codeShell:\n", codeShell)
        
        archiveExample = self.ExampleSelection("inputOutputArchive", lang)
        ArchivePrompt= self.promptSelect.inputOutputArchive(lang, All_MS+"\n...\n"+MainFunc, ExecuteOutput.strip("\n"), archiveExample.strip("\n"))
        print("ArchivePrompt\n",ArchivePrompt)
        if self.debug: ArchiveResponse = self.model.generate(ArchivePrompt, False,True)
        else: ArchiveResponse = self.chatgptModel.generate(ArchivePrompt, False,True)
        
        print("ArchiveResponse:\n", ArchiveResponse)

        
        Input_and_ExpectedOutput = self.TestInput_responseDeal(ArchiveResponse).replace("\n----\n","\n--------\n")
        print("codeShell_IO_Gen_Input_and_ExpectedOutput:\n",Input_and_ExpectedOutput)
        return codeShell, Input_and_ExpectedOutput, Input_and_ExpectedOutput.split("--------")[:3]
    
    def IO2CodeShell_Gen(self, fileName, function_code, ImportInfo, All_MS, method_signature, lang, Input_and_ExpectedOutput):
        
        MianGenExample = self.ExampleSelection("MainGeneration", lang)
        MS_code = All_MS + "\n" + " ..."
        
        pattern = re.compile(r'output\d+:', re.IGNORECASE)
        TestInput_refined = "\n".join([re.split(pattern, groupVar)[0].strip() for groupVar in Input_and_ExpectedOutput.split("--------")])
        target_MainFuncGenPrompt = self.promptSelect.MainFuncGen(lang, function_code, TestInput_refined, method_signature, MianGenExample.strip('\n'))
        print("IO2CodeShell_Gen_target_MainFuncGenPrompt:\n",target_MainFuncGenPrompt)
        
        if self.debug: target_MainGen_response = self.model.generate(target_MainFuncGenPrompt, False)
        else: target_MainGen_response = self.chatgptModel.generate(target_MainFuncGenPrompt, False)
        
        print("target_MainGen_response: \n",target_MainGen_response)
        target_MainFunc, import_info = self.MainFunc_responseDeal(target_MainGen_response, lang)  # 从 response 当中得到 main function
        
        target_CodeCont, target_codeShell = self.GetCodeCont(target_MainFunc.strip(), function_code, ImportInfo.strip() + "\n" + import_info, fileName.split(".")[0], lang)
        _, stderroutput,_ = self.codeRun(fileName, target_CodeCont, lang)
        if len(stderroutput) != 0: 
            print("IO2CodeShell_Gen_stderroutput:\n",stderroutput)
            return "runFailed"
        return target_codeShell


    def MainGene(self, functionCode, lang, TestInputs, method_signature):
        MainGenExample = self.ExampleSelection("MainGeneration",lang)
        MainFuncGenPrompt = self.promptSelect.MainFuncGen(lang, functionCode,TestInputs.strip("\n"), method_signature, MainGenExample)
        print("MainFuncGenPrompt:\n", MainFuncGenPrompt)
        
        if self.debug: MainGen_response = self.model.generate(MainFuncGenPrompt, False)
        else: MainGen_response = self.chatgptModel.generate(MainFuncGenPrompt, False,True)
        
        print("MainGen_response:\n", MainGen_response)
        
        MainFunc, import_info = self.MainFunc_responseDeal(MainGen_response, lang)  
        return MainFunc, import_info
    
    def TestPrune(self, fileName, CodeCont, lang):
        iterate = 0
        ExecuteOutput = ""
        while True:
            ExecuteOutput, stderroutput, Tag = self.codeRun(fileName, CodeCont, lang)
            if len(stderroutput) == 0 or "TimeoutExpired" == stderroutput:  break
            print("TestPrune_stderroutput:\n",stderroutput)
            print("TestPrune_Tag:\n",Tag)
            
            CodeCont = self.ErrorInfoDeal(stderroutput, fileName, CodeCont, lang, Tag)
            iterate = iterate + 1
            if iterate >= 5: 
                print("ERROR: ",fileName)
                ExecuteOutput = ""
                break
        return ExecuteOutput, CodeCont
                    

        
    def GetCodeCont(self, MainFunc, functionCode, importInfo, fileName, lang):
        if lang == "python":
            codeShell = importInfo + "\n" + "# TOFILL" + "\n" + MainFunc
            allCode = codeShell.replace("# TOFILL",functionCode)
        elif lang == "java":
            codeShell =importInfo + "\n" + f"public class {fileName} {{\n" + "// TOFILL" + "\n" + MainFunc + "\n}"
            allCode = codeShell.replace("// TOFILL",functionCode)
        elif lang == "cpp":
            codeShell = importInfo + "\n" + "// TOFILL" + "\n" + MainFunc
            allCode = codeShell.replace("// TOFILL",functionCode)
        return allCode.strip(), codeShell.strip()
    
    def codeRun(self, fileName, CodeCont, lang):
        fileWritePath = f"./CodeTranslation/CodeTransInputOutput/gen_temp/{self.source_lang}2{self.target_lang}/{self.source_lang}"
        self.boolean(fileWritePath)
        targetCodePath = os.path.join(fileWritePath, fileName)
        with open(targetCodePath,'w',encoding='utf-8') as f:
            f.write(CodeCont)
        
        if lang == "python":
            run_result = subprocess.run(f"python {targetCodePath}", shell=True, capture_output=True, text=True, timeout=10)
            stdoutput = run_result.stdout
            stderroutput = run_result.stderr
            return stdoutput, stderroutput, "Compile"
        elif lang == "java":
            java_home = "./JavaEnv/jdk-17.0.11"
            javafx_path = "./JavaEnv/javafx-sdk-17.0.9/lib/*.jar"
            javafx_jars = glob.glob(javafx_path) 
            javafx_classpath = os.pathsep.join(javafx_jars)
            javac_command = f"{java_home}/bin/javac -cp \"{javafx_classpath}\" {targetCodePath}"
            return_result = subprocess.run(javac_command, capture_output=True, shell=True, timeout=10)
            
            if return_result.returncode != 0: return return_result.stdout.decode('utf-8'), return_result.stderr.decode('utf-8'), "Compile"
            try:
                os.chdir(os.path.dirname(targetCodePath))
                class_name = os.path.splitext(os.path.basename(targetCodePath))[0]
                java_command = f"{java_home}/bin/java -cp {javafx_classpath}:{os.path.dirname(targetCodePath)} {class_name}"
                run_result = subprocess.run(java_command, shell=True, capture_output=True, text=True, timeout=10)
                os.chdir(current_dir)
                stdoutput = run_result.stdout
                stderroutput = run_result.stderr
                return stdoutput, stderroutput, "Run"
            
            except subprocess.TimeoutExpired as e:
                os.chdir(current_dir)
                output_str = e.stdout.decode('utf-8') if e.stdout else ""
                return output_str, "TimeoutExpired","TimeoutExpired"
            
        elif lang == "cpp":
            output_executable =  os.path.splitext(os.path.basename(targetCodePath))[0]
            directory = os.path.dirname(targetCodePath)
            include_dir = "./JavaEnv/cpp_nlohmann/json/include"
            compile_command = ["g++", "-Wall", "-g","-I", include_dir, targetCodePath, "-o", os.path.join(directory, output_executable)]
            
            compile_result = subprocess.run(compile_command, capture_output=True, text=True, timeout=10)
            if compile_result.returncode != 0: return compile_result.stdout, compile_result.stderr, "Compile"
            try:
                gdb_command = ["gdb", "--batch", "-ex", "run", "-ex", "bt", os.path.join(directory, output_executable)]
                run_result = subprocess.run(gdb_command, capture_output=True, text=True, timeout=10)
                if "No stack." in run_result.stderr: 
                    return run_result.stdout, "", "Run"
                else: 
                    return "", run_result.stdout, "Run"
                
            except subprocess.TimeoutExpired as e:
                output_str = e.stdout.decode('utf-8') if e.stdout else ""
                return output_str, "TimeoutExpired","TimeoutExpired"

    def ErrorInfoDeal(self, ErrorInfo, fileName, CodeCont, lang, Tag):
        ErrorInfo_list = [error for error in ErrorInfo.split("\n") if error.strip()]
        if lang == "cpp": pattern = r"\.cpp:(\d+)"
        elif lang == "java": pattern = r"\.java:(\d+)"
        elif lang == "python": pattern = r", line (\d+)"
        
        if Tag == "Compile": ErrorInfo_list = ErrorInfo_list
        elif Tag == "Run": ErrorInfo_list = reversed(ErrorInfo_list)
        
        for errorInfo in ErrorInfo_list:
            if (fileName in errorInfo and "error:" in errorInfo) or (fileName.split(".")[0] in errorInfo):
                match = re.search(pattern, errorInfo)
                Error_Line = int(match.group(1))
                break
            
        new_CodeCont = []
        CodeCont_list = CodeCont.split("\n")
        for i in range(len(CodeCont_list)):
            if i == Error_Line-1: continue
            new_CodeCont.append(CodeCont_list[i])
        return "\n".join(new_CodeCont)
        
        
    def ExampleSelection(self, strategy, lang):
        if "TestGeneration" in strategy:
            if "python" == lang:
                Example = handcraftPrompt.TestInputGen_python
            elif "java" == lang:
                Example = handcraftPrompt.TestInputGen_java
            elif "cpp" == lang:
                Example = handcraftPrompt.TestInputGen_cpp
            return Example
        elif "MainGeneration" in strategy:
            if "python" == lang:
                Example = handcraftPrompt.MainFuncGen_python
            elif "java" == lang:
                Example = handcraftPrompt.MainFuncGen_java
            elif "cpp" == lang:
                Example = handcraftPrompt.MainFuncGen_cpp
            return Example
        elif "inputOutputArchive" in strategy:
            if "java" == lang:
                Example = handcraftPrompt.inputOutputArchive_java
            elif "cpp" == lang:
                Example = handcraftPrompt.inputOutputArchive_cpp
            elif "python" == lang:
                Example = handcraftPrompt.inputOutputArchive_python
            return Example
        
            
    def ValueRefactor(self, MainFunc, stdoutput, methodSig):
        
        ExpectedOutput_list = [out for out in stdoutput.split("\n") if out.strip()]
        if self.source_lang == "python":
            MainFunc_list = [printStmt for printStmt in MainFunc.split("\n") if printStmt.strip() and "print" in printStmt]
        elif self.source_lang == "java":
            MainFunc_list = [printStmt for printStmt in MainFunc.split("\n") if printStmt.strip() and "System.out.print" in printStmt]
        elif self.source_lang == "cpp":
            MainFunc_list = [printStmt for printStmt in MainFunc.split("\n") if printStmt.strip() and "std::cout" in printStmt]
        if len(ExpectedOutput_list) != len(MainFunc_list): return "", ""
        
        Input_list = []
        for printStmt in MainFunc_list:
            KeyIndex = self.toolUse.indexFinder([methodSig],printStmt)
            methodSig_param = self.toolUse.BracketFinder(KeyIndex, printStmt)
            Input_list.append(methodSig_param)
            
        Input_and_Output = []
        for index in range(len(ExpectedOutput_list)):
            Input_and_Output.append([Input_list[index], ExpectedOutput_list[index]])
        return Input_and_Output


    def fileWrite(self, Data, Gen_outPath_dir):
        source_code = Data['source_code_str']
        if self.dataset_name == "transCoder_st":
            if self.source_lang == "python":
                pythonCode = source_code.replace("f_gold","f_filled")
                writenCode = Data['source_code_shell'].replace("# TOFILL", pythonCode)
            elif self.source_lang == "java":
                JavaCode =source_code.replace("f_gold","f_filled")
                writenCode =  Data['source_code_shell'].replace("// TOFILL", JavaCode)
            genefilePath = os.path.join(Gen_outPath_dir,Data['source_Lan'].split("###")[-1])
            with open(genefilePath, 'w', encoding='utf-8') as f:
                f.write(writenCode)
        elif self.dataset_name == "codefuse":
            if self.source_lang == "python":
                pythonCode = source_code
                writenCode = Data['source_code_shell'].replace("# TOFILL", pythonCode)
            elif self.source_lang == "java":
                JavaCode = source_code
                writenCode =  Data['source_code_shell'].replace("// TOFILL", JavaCode)
            genefilePath = os.path.join(Gen_outPath_dir,Data['source_Lan'].split("###")[-1])
            with open(genefilePath, 'w', encoding='utf-8') as f:
                f.write(writenCode)
        return genefilePath

    def find_original_index(self,str1, sub_str1):
        normalized_sub_str1 = re.sub(r'\s+', ' ', sub_str1.strip())
        
        normalized_str1 = ''
        index_map = []
        last_char = None
        for index, char in enumerate(str1):
            if char.isspace():
                if last_char and not last_char.isspace():
                    normalized_str1 += ' '
                    index_map.append(index)
            else:
                normalized_str1 += char
                index_map.append(index)
            last_char = char

        pattern = re.escape(normalized_sub_str1)
        match = re.search(pattern, normalized_str1)
        if match:
            return index_map[match.start()], index_map[match.end()]
        else:
            return -1, -1  # 如果没有找到，返回-1
    
    def count_spaces_before_substring(self, str1, sub_str1):
        start_index = str1.find(sub_str1)
        if start_index == -1:
            return -1
        count = 0
        for i in range(start_index - 1, -1, -1):
            if str1[i] == ' ':
                count += 1
            else:
                break
        return count
    
    def TestInput_responseDeal(self, response):
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if not matches: return response.split("## Output")[-1].replace("```","")
        math_temp = [match for match in matches if match.strip()]
        if len(math_temp): return math_temp[-1]
        else: return response 

    def MainFunc_responseDeal(self, response, lang):
        pattern = r"```(.*?)```"
        matches_t = re.findall(pattern, response, re.DOTALL)
        if not matches_t: 
            matchCode = re.sub(r'(python|java|cpp)\n', '', response, flags=re.IGNORECASE)
        else: 
            matches = [match for match in matches_t if len(match)>10 and "main" in match ]
            matchCode = re.sub(r'(python|java|cpp)\n', '', matches[-1], flags=re.IGNORECASE)
        MainFunc = ""
        import_info = "\n".join([code for code in matchCode.split("\n") if "import " in code or "#include" in code or "# include" in code])
        if lang == "python":
            MainFunc = self.extract_Mainfunction_python(matchCode)
        elif lang == "java" or lang == "cpp":
            MainFunc = self.extract_Mainfunction_java_cpp(matchCode)
        return MainFunc, import_info

    def extract_Mainfunction_python(self, Code):
        codeList = [code for code in Code.split("\n") if code.strip()]
        Method_Signature = ""
        functionBlock  = []
        StartTag = False
        Indent = -1
        for code in codeList:
            if ("if __name__" in code or ("def" in code and "main" in code.lower())) and StartTag==False:
                functionBlock.append('''if __name__ == "__main__":''')
                StartTag = True
                Indent = self.GetIndent(code)
                continue
            if StartTag and Indent !=-1:
                Indent_temp = self.GetIndent(code)
                if Indent_temp <= Indent:
                    break
                functionBlock.append(code)
        return "\n".join(functionBlock)
    
    def extract_Mainfunction_java_cpp(self,Code):
        Code = Code.replace(" (","(")
        Code_list = Code.split("\n")
        codeBlock = []
        left_brack_list = []
        right_brack_list = []
        Start_Tag = False
        for current_line_number, line in enumerate(Code_list):
            if " main(" in line and "*" not in line:
                Start_Tag = True
                codeBlock.append(line)
                left_brack_count = line.count("{")
                left_brack_list.extend(["{"] * left_brack_count)
                right_brack_count = line.count("}")
                right_brack_list.extend(["}"] * right_brack_count)
                if len(left_brack_list) == len(right_brack_list): break
                continue
            if Start_Tag:
                codeBlock.append(line)
                
                left_brack_count = line.count("{")
                left_brack_list.extend(["{"] * left_brack_count)
                right_brack_count = line.count("}")
                right_brack_list.extend(["}"] * right_brack_count)
                if len(left_brack_list) == len(right_brack_list):
                    break
        functionBlock = "\n".join(codeBlock)
        
        return functionBlock
    
         
    def boolean(self,file_path):
        if not os.path.exists(file_path):
            print('Creat floder....')
            os.makedirs(file_path)
        else:
            shutil.rmtree(file_path)
            os.makedirs(file_path)

    def GetIndent(self, s):
        return len(s) - len(s.lstrip(' '))  
    
                    
                  
if __name__ == "__main__":
    
    import sys
    args = sys.argv[1:]
    tasks = []
    if len(args) >= 1:
        tasks.append(args[0])
    if len(args) >= 2:
        tasks.append(args[1])
    
    model = 'gpt'
    strategys = ['TestGeneration'] 
    dataset_names = ['manually']
    tasks = ["java2cpp","java2python","python2cpp","python2java","cpp2java","cpp2python"]
    ValidTest = True
    for strategy in strategys:
        print("strategy:", strategy)
        for dataset_name in dataset_names:
            print("dataset_name:", dataset_name)
            for task in tasks:
                print("task:", task)
                source_lang = task.split("2")[0] 
                target_lang = task.split("2")[-1]
                if ValidTest: output_path = os.path.join("./CodeTranslation/CodeTransInputOutput","manually")  
                else: output_path = os.path.join("./CodeTranslation/CodeTransInputOutput",dataset_name)  
                TestGenMain(source_lang, target_lang, model, strategy, dataset_name, ValidTest)