import torch
import transformers
from transformers import LlamaForCausalLM,LlamaTokenizer,AutoModel, AutoModelForCausalLM,AutoTokenizer
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
from Dataloader import transCoder_st, codeFuse
from Compile_and_Run_testInputs import Compile_and_Run
from definedTool import definedTool
import BasicBlock

current_dir = os.getcwd()
CodeTrans_path = os.path.normpath(os.path.join(current_dir, '../'))
import subprocess


class HuggingfaceModel:
    def __init__(self,model_name_or_path,model_type):
        self.model_type = model_type
    
        if "llama3_0_8B" in model_type:
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            
        elif "chatglm2_6b" in model_type:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).cuda()
            self.model = model.eval()

        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.float16, low_cpu_mem_usage=True).to('cuda')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            
    
    def generate(self,query,return_input=False,skip_special_tokens = True, do_sample=False,temperature = 0, max_new_tokens = 1024)->str:
        if "deepseek" in self.model_type:
            messages=[
                    { 'role': 'user', 'content': query}
                ]
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('cuda')
            # 32021 is the id of <|EOT|> token
            outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,temperature =temperature, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
            
            return self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        elif "llama3_0_8B" in self.model_type:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to('cuda')
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ] 
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=do_sample,
                temperature=temperature,
            )
            response = outputs[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(response, skip_special_tokens=True)

        elif "chatglm2_6b" in self.model_type:
            response, history = self.model.chat(self.tokenizer, query, max_new_tokens=max_new_tokens, history=[])
            
            return response
            
        else:
            query = self.process_prompt(query)
            # query = "This is a test"
            input_ids = self.tokenizer.encode(query,return_tensors='pt').to('cuda')
            input_length = len(query)
            tokens = self.model.generate(input_ids,max_new_tokens=2048, do_sample=False,temperature = 0)
            response = self.tokenizer.decode(tokens[0],skip_special_tokens=skip_special_tokens)
            if return_input==True:
                return response
            else:
                return response[input_length:]
    
    def process_prompt(self,query):
        if 'codellama' in self.model_type:
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
            _system_prompt = '''
            You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            '''
            system_prompt = f"{B_SYS}{_system_prompt}{E_SYS}"
            prompt = system_prompt+f"{B_INST} {query.strip()} {E_INST} "
            return prompt

class ChatGenerationModel:
    def __init__(self):
        OPENAI_API_KEY = 'KEY'
        openai.api_key = OPENAI_API_KEY
        openai.api_base = 'https://ai.devtool.tech/proxy/v1'
        self.model_name = 'gpt-3.5-turbo'

    
    def generate(self,query,return_input=False,skip_special_tokens = True)->str:
        messages = [
                {"role":"user","content":query}
            ]
        response = openai.ChatCompletion.create(
            model = self.model_name,
            messages = messages,
            max_tokens = 1536,
            n=1
        )
        answer = response.choices[0]['message']['content']

        return answer

class MockModel:
    def generate(self,query,return_input=False):
        return "111111111111111111111111"


class PromptDefine:
    def Code2CodeTrans(self, source_lang,target_lang, To_be_translated, Method_signature, source_example, target_example, Method_signature_example):
        t_template = PromptTemplate(template=handcraftPrompt.Code2CodeTrans_trans,input_variables=['source_Lan','target_Lan', 'To_be_translated', 'Method_signature', 'source_example', 'target_example', 'Method_signature_example'])
        template = t_template.format(source_Lan=source_lang.capitalize(), target_Lan =target_lang.capitalize(), To_be_translated = To_be_translated, Method_signature=Method_signature, source_example=source_example, target_example=target_example, Method_signature_example = Method_signature_example)
        return template
    
    def Code2CodeTrans_inputOutput(self, source_lang,target_lang, To_be_translated, Method_signature, Input_output, source_example,Input_output_example,Method_signature_example,target_example):
        t_template = PromptTemplate(template=handcraftPrompt.Code2CodeTrans_trans_inputOutput,input_variables=['source_Lan','target_Lan', 'To_be_translated', 'Method_signature', 'Input_output','source_example','Input_output_example','Method_signature_example','target_example'])
        template = t_template.format(source_Lan=source_lang.capitalize(), target_Lan =target_lang.capitalize(), To_be_translated = To_be_translated, Method_signature=Method_signature, Input_output=Input_output, source_example=source_example,Input_output_example=Input_output_example,Method_signature_example=Method_signature_example,target_example=target_example)
        return template
    
    def TypeInference(self,source_lang, source_code, target_lang, TestCase, Example):
        t_template = PromptTemplate(template=handcraftPrompt.VariableTypeInference,input_variables=['source_lang', 'source_code', 'target_lang', 'TestCase', 'Example'])
        template = t_template.format(target_lang =target_lang.capitalize(), source_lang =source_lang.capitalize(), TestCase=TestCase, source_code=source_code, Example = Example)
        return template    
    
    def CodeCompile_Repair(self, target_lang, target_code,Method_signature, TestCase, ErrorMessage, commentTag, Example):
        t_template = PromptTemplate(template=handcraftPrompt.Code2CodeTrans_trans_inputOutput_Compilerepair,input_variables=[ 'target_lang', 'TestCase', 'ErrorMessage','target_code','commentTag','Method_signature', 'Example'])
        template = t_template.format(target_lang =target_lang.capitalize(), TestCase=TestCase, ErrorMessage=ErrorMessage, target_code=target_code, commentTag = commentTag, Method_signature = Method_signature, Example = Example)
        return template
    
    def CodeCompile_Repair_our(self, target_lang, target_code,Method_signature, TestCase, ErrorMessage, commentTag, Example):
        t_template = PromptTemplate(template=handcraftPrompt.Code2CodeTrans_trans_inputOutput_Compilerepair_our,input_variables=[ 'target_lang', 'TestCase', 'ErrorMessage','target_code','commentTag','Method_signature','Example'])
        template = t_template.format(target_lang =target_lang.capitalize(), TestCase=TestCase, ErrorMessage=ErrorMessage, target_code=target_code, commentTag = commentTag, Method_signature = Method_signature, Example = Example)
        return template
    def CodeRun_RunTimerepair(self, target_lang, target_code, TestCase, Line, ErrorMessage, Method_signature, Example):
        t_template = PromptTemplate(template=handcraftPrompt.Code2CodeTrans_inputOutput_RunTimerepair,input_variables=['target_lang', 'target_code', 'TestCase', 'Line', 'ErrorMessage', 'Method_signature', 'Example'])
        template = t_template.format(target_lang =target_lang.capitalize(), target_code=target_code, TestCase=TestCase, Line=Line, ErrorMessage=ErrorMessage, Method_signature=Method_signature, Example = Example)
        return template
    
    def CodeRun_AllInRunrepair(self, target_lang, target_code, TestCase, ErrorMessage, Method_signature, Example):
        t_template = PromptTemplate(template=handcraftPrompt.Code2CodeTrans_inputOutput_AllInRunrepair,input_variables=['target_lang', 'target_code', 'TestCase', 'ErrorMessage', 'Method_signature','Example'])
        template = t_template.format(target_lang=target_lang.capitalize(), target_code=target_code, TestCase=TestCase, ErrorMessage=ErrorMessage, Method_signature=Method_signature, Example=Example)
        return template
    def CodeMapping(self, source_lang,To_be_translated, target_lang, target_code, Example, commentTag):
        t_template = PromptTemplate(template=handcraftPrompt.CodeMapping,input_variables=['source_lang','To_be_translated', 'target_lang', 'target_code', 'Example', 'commentTag'])
        template = t_template.format(source_lang=source_lang.capitalize(),target_lang=target_lang.capitalize(), target_code=target_code, To_be_translated=To_be_translated, Example=Example, commentTag=commentTag)
        return template

    def TransMapping(self, source_lang,To_be_translated, target_lang, target_code, Example):
        t_template = PromptTemplate(template=handcraftPrompt.transMap,input_variables=['source_lang','To_be_translated', 'target_lang', 'target_code', 'Example'])
        template = t_template.format(source_lang=source_lang.capitalize(),target_lang=target_lang.capitalize(), target_code=target_code, To_be_translated=To_be_translated, Example=Example)
        return template
    
    def FunctionDescription(self, source_lang, target_lang, Code, Example):
        t_template = PromptTemplate(template=handcraftPrompt.FunctionDescription,input_variables=['source_lang','target_lang', 'Code', 'Example' ])
        template = t_template.format(source_lang=source_lang.capitalize(),target_lang=target_lang.capitalize(), Code = Code, Example = Example)
        return template

    def TestFG(self,source_lang, target_lang, source_block_code, target_code, Example):
        t_template = PromptTemplate(template=handcraftPrompt.TestFG,input_variables=['source_lang', 'target_lang', 'source_block_code', 'target_code', 'Example'])
        template = t_template.format(source_lang=source_lang.capitalize(),target_lang=target_lang.capitalize(), source_block_code=source_block_code, target_code=target_code, Example=Example)
        return template

    
    def RGCode(self, Lang, Code, Example):
        t_template = PromptTemplate(template=handcraftPrompt.RGCode,input_variables=['Lang', 'Code', 'Example' ])
        template = t_template.format(Lang=Lang.capitalize(), Code = Code, Example = Example)
        return template
    
    def selfRefine(self,source_lang, target_lang, source_code, trans_code):
        t_template = PromptTemplate(template=handcraftPrompt.selfRefine,input_variables=['source_lang', 'target_lang', 'source_code', 'trans_code'])
        template = t_template.format(source_lang=source_lang.capitalize(),target_lang=target_lang.capitalize(), source_code=source_code, trans_code=trans_code)
        return template

    def TestFG_Refine(self,source_lang, target_lang, source_code, trans_code, Input_dict, Actually_Output_Expected_Output,Example):
        Actually_Output = Actually_Output_Expected_Output.split("$$$$")[0]
        Expected_Output = Actually_Output_Expected_Output.split("$$$$")[1]
        if source_lang == "cpp": source_lang = "c++"
        if target_lang == "cpp": target_lang = "c++"
        t_template = PromptTemplate(template=handcraftPrompt.TestFG_Refine,input_variables=['source_lang', 'target_lang', 'source_code', 'trans_code', 'Input_dict', 'Actually_Output', 'Expected_Output'])
        template = t_template.format(source_lang=source_lang.capitalize(),target_lang=target_lang.capitalize(), source_code=source_code, trans_code=trans_code, Input_dict=Input_dict, Actually_Output=Actually_Output, Expected_Output=Expected_Output)
        return template
    
    def TestFG_Further(self,source_lang, target_lang, source_code, trans_code, Input_dict, Actually_Output_Expected_Output,Example):
        # Actually_Output = Actually_Output_Expected_Output.split("$$$$")[0]
        # Expected_Output = Actually_Output_Expected_Output.split("$$$$")[1]
        if source_lang == "cpp": source_lang = "c++"
        if target_lang == "cpp": target_lang = "c++"
        t_template = PromptTemplate(template=handcraftPrompt.TestFG_FurtherRefine,input_variables=['source_lang', 'target_lang', 'source_code', 'trans_code'])
        template = t_template.format(source_lang=source_lang.capitalize(),target_lang=target_lang.capitalize(), source_code=source_code, trans_code=trans_code)
        
        return template
     
    def MessageConver(self, ErrorMessage, ErrorLine, ParameterError, Example, function_code, lang, model='others'):
        if ParameterError:
            t_template = PromptTemplate(template=handcraftPrompt.MessageConver_parameterType,input_variables=['ErrorMessage', 'ErrorLine','Example' ])
            template = t_template.format(ErrorMessage = ErrorMessage, ErrorLine = ErrorLine, Example = Example)
            if model == "chatglm2_6b":
                t_template = PromptTemplate(template=handcraftPrompt.MessageConver_parameterType_llama2,input_variables=['ErrorMessage', 'ErrorLine','Example' ])
                template = t_template.format(ErrorMessage = ErrorMessage, ErrorLine = ErrorLine, Example = Example)
        else:
            t_template = PromptTemplate(template=handcraftPrompt.MessageConver_not_parameterType,input_variables=['ErrorMessage', 'ErrorLine', 'function_code','lang','Example'])
            template = t_template.format(ErrorMessage = ErrorMessage, ErrorLine = ErrorLine, function_code = function_code,lang=lang.capitalize(), Example = Example)
            if model == "chatglm2_6b":
                t_template = PromptTemplate(template=handcraftPrompt.MessageConver_not_parameterType_llama2,input_variables=['ErrorMessage', 'ErrorLine', 'function_code','lang','Example'])
                template = t_template.format(ErrorMessage = ErrorMessage, ErrorLine = ErrorLine, function_code = function_code,lang=lang.capitalize(), Example = Example)
        return template 
    
    def selfAnalyzeFG(self, source_lang, source_code, target_lang, target_code, Input_dict, Actually_Expected_output):
        Actually_Expected_output = Actually_Expected_output.replace("$LANG$", f"`{target_lang.capitalize()}_code`")
        t_template = PromptTemplate(template=handcraftPrompt.selfAnalyzeFG,input_variables=['source_lang', 'source_code', 'target_lang', 'target_code', 'Input_dict', 'Actually_output', 'Actually_Expected_output'])
        template = t_template.format(source_lang=source_lang.capitalize(), source_code=source_code, target_lang=target_lang.capitalize(), target_code=target_code, Input_dict=Input_dict, Actually_Expected_output=Actually_Expected_output)
        return template
    
    def VarUpdate(self, source_lang, source_code, Example):
        t_template = PromptTemplate(template=handcraftPrompt.VarUpdate,input_variables=['source_lang', 'source_code', 'Example' ])
        template = t_template.format(source_lang=source_lang.capitalize(), source_code = source_code, Example = Example)
        return template
    
class Main:
    def __init__(self,source_lang,target_lang, model_name,strategy, dataset_name, isOur):
        if model_name == 'deepseek_7B':
            self.model = HuggingfaceModel("./huggingface/hub/deepseek-coder-6.7b-instruct" ,model_name)
        elif model_name == "llama3_0_8B":
            self.model = HuggingfaceModel("./huggingface/hub/Meta-Llama-3-8B-Instruct", model_name) 
        elif model_name == "chatglm2_6b":
            self.model = HuggingfaceModel("./huggingface/hub/chatglm2-6b",model_name)
        self.model_type = model_name
        self.isOur = isOur
        self.promptSelect= PromptDefine()
        self.toolUse = definedTool()
        self.dataset_name = dataset_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.allMS = False
        
        # dataloader selection
        if strategy == "inputOutput":
            DataList = []
            fileName = f"{self.source_lang}2{self.target_lang}_data.jsonl"
            inputDir = "./CodeTranslation/CodeTranslation/CodeTransInputOutput"
            
            inputDataPath = os.path.join(inputDir,self.dataset_name,fileName)
            with open(inputDataPath, 'r', encoding='utf-8') as f:
                for line in f:
                    cont = json.loads(line.strip())
                    if self.source_lang == "python": cont['source_code_str'] = self.toolUse.remove_comments_python(cont['source_code_str'])
                    else: cont['source_code_str'] = self.toolUse.remove_comments_java_cpp(cont['source_code_str'])
                    if ("TAG" in cont and "ERROR_" in cont['TAG']) or "input_output_value" not in cont or "runFailed" in cont['input_output_value'] or len(cont['input_output_value']) == 0 or 'target_code_shell' not in cont or "runFailed" in cont['target_code_shell'] or "source_code_shell" not in cont or  "runFailed" in cont['source_code_shell']: continue
                    DataList.append(cont)
                    
        elif "compilationCheck" in strategy:
            DataList = []
            fileName = model_name + "_" + strategy.split("_")[0] + ".jsonl"
            inputDataPath = os.path.join(output_path,fileName)
            with open(inputDataPath, 'r', encoding='utf-8') as f:
                for line in f:
                    cont = json.loads(line.strip())
                    DataList.append(cont)
                    
        elif "runCheck" in strategy:
            DataList = []
            if "FSE" not in strategy and "runCheckTestFG_Further" not in strategy:
                fileName = model_name + "_" + strategy.split("_")[0] + "_compilationCheck_our.jsonl" 
            elif "runCheckTestFG_Further" in strategy:
                fileName = f"{model_name}_inputOutput_runCheckTestFG_Refine.jsonl"
            else:
                fileName = model_name + "_" + strategy.split("_")[0] + "_compilationCheck_fse.jsonl" 
            inputDataPath = os.path.join(output_path,fileName)
            with open(inputDataPath, 'r', encoding='utf-8') as f:
                for line in f:
                    cont = json.loads(line.strip())
                    DataList.append(cont)
        
        elif "_alignment" in strategy:
            DataList = []
            inputDataPath = f"./CodeTranslation/CodeTranslation/codeAlignment/{self.source_lang}2{self.target_lang}.jsonl"
            with open(inputDataPath, 'r', encoding='utf-8') as f:
                for line in f:
                    cont = json.loads(line.strip())
                    DataList.append(cont)   
        
        final_result_path = os.path.join(output_path,model_name+"_"+strategy+".jsonl")       
        self.mainStart(DataList,strategy,final_result_path)
    
    def mainStart(self, DataList,strategy,final_result_path):
        file_cont = []
        if os.path.exists(final_result_path):
            with open(final_result_path, 'r', encoding='utf-8') as f:
                for line in f:
                    cont = json.loads(line.strip())
                    file_cont.append(cont['source_Lan'])
                      
        if strategy == "inputOutput":
            compileRun = Compile_and_Run(self.target_lang, self.dataset_name,strategy)
            for Data in tqdm(DataList):
                print("Deal: ", Data["source_Lan"])
                
                if Data["source_Lan"] in file_cont: continue
                
                Method_signature = Data['target_method_signature'].strip()
                target_allMS = Method_signature
                
                To_be_translated = Data['source_code_str']
                source_example, target_example, Method_signature_example = self.ExampleSelection("inputOutput")
                Input_output_example = handcraftPrompt.Input_output_example
                
                
                t_prompt = self.promptSelect.Code2CodeTrans_inputOutput(self.source_lang, self.target_lang, To_be_translated.strip('\n'), target_allMS.strip(), "".join(Data['input_output_value']).strip("\n"),source_example.strip('\n'),Input_output_example.strip('\n'),Method_signature_example,target_example.strip('\n'))
                print(t_prompt)
                
                response = self.model.generate(t_prompt,False)
                print("response:\n", response)
                import_info, translated_code = self.responseDeal(response, self.target_lang, Method_signature)
                print("translated_code:\n", translated_code)
                if len(translated_code) == 0: 
                    Data['testResult'] = "NoMatchCode"
                    with open(final_result_path, 'a', encoding='utf-8') as f:
                        json.dump(Data,f)
                        f.write('\n')   
                    continue
                
                Data['target_code_shell'] = import_info + "\n" + Data['target_code_shell']
                Data['trans_code'] = translated_code 
                Data['prompt'] = t_prompt

                Error_Numbers, ErrorMessageFunction, _, _ = compileRun.mainStart(Data, "compileRepair",boolStrategy=True)
                if Error_Numbers == 0:
                    Data['compileResult'] = 1
                    STATE,Group, blockNumber, InputOutput_function, _, _, _, _,_ = compileRun.mainStart(Data, "runRepair",boolStrategy=True)
                    if len(InputOutput_function)==0:
                        testResult = 1
                    elif InputOutput_function in ["NOTEQUAL","INST_ERROR"]: testResult = InputOutput_function
                    else: testResult = 0
                    
                    Data['testResult'] = testResult
                else: 
                    Data['compileResult'] = 0
                    Data['testResult'] = 0
                    
                with open(final_result_path, 'a', encoding='utf-8') as f:
                    json.dump(Data,f)
                    f.write('\n')   
                                       
        elif "compilationCheck_fse" in strategy :
            Example = self.ExampleSelection(strategy)
            
            compileRun = Compile_and_Run(self.target_lang, self.dataset_name, "inputOutput")
            for Data in tqdm(DataList):
                if Data["source_Lan"] in file_cont: continue
                if 'trans_code' not in Data or len(Data['trans_code'])==0: continue
                print("####",Data['source_Lan'])
                
                Method_signature = Data['target_method_signature'].strip()
                
                target_allMS = Method_signature
                if not Data['compileResult'] == 0 or Data['testResult']== "NOTEQUAL": 
                    with open(final_result_path, 'a', encoding='utf-8') as f:
                        json.dump(Data,f)
                        f.write('\n')    
                    continue


                Error_Numbers, ErrorMessageFunction, TestCase, ErrorMessage = compileRun.mainStart(Data, "compileRepair", isOur=self.isOur)
                error_number_list = []
                error_log = [""]
                iterative_count = 0
                import_info = ""
                while Error_Numbers and iterative_count <= 3: 
                    error_log.append(ErrorMessageFunction)
                    error_number_list.append(Error_Numbers)
                    if "<Buggy Line>" not in ErrorMessageFunction:
                        translated_code = ErrorMessageFunction
                        t_prompt = ""
                    else:
                        t_prompt = self.promptSelect.CodeCompile_Repair(target_lang, ErrorMessageFunction.strip("\n"), target_allMS.strip(), TestCase, ErrorMessage, Data["commentTag"], Example)
                        print(t_prompt)
                        response = self.model.generate(t_prompt,False)
                        import_info, translated_code = self.responseDeal(response, target_lang, Method_signature)
                    if len(translated_code) == 0: break    
                    Data['trans_code'] = translated_code.strip()
                    Data['target_code_shell'] = import_info + "\n" + Data['target_code_shell']
                    Data['prompt'] = t_prompt
                    Error_Numbers, ErrorMessageFunction, TestCase, ErrorMessage = compileRun.mainStart(Data, "compileRepair",isOur=self.isOur, boolStrategy=True)
                    iterative_count = iterative_count + 1
                if iterative_count == 0: Data['prompt'] = "" 
                    
                if Error_Numbers == 0:
                    compileResult = 1

                    STATE, Group, blockNumber, InputOutput_function, _, _, _,_,_ = compileRun.mainStart(Data, "runRepair",boolStrategy=True)
                    if len(InputOutput_function)==0:
                        testResult = 1
                    elif InputOutput_function in ["NOTEQUAL","INST_ERROR"]: testResult = InputOutput_function
                    else: testResult = 0

                else:
                    compileResult = 0
                    testResult = 0
                    
                Data['compileResult'] =  compileResult
                Data['iterativeCount'] =  iterative_count
                Data['testResult'] = testResult
                Data['import_info'] = import_info
                with open(final_result_path, 'a', encoding='utf-8') as f:
                    json.dump(Data,f)
                    f.write('\n')  

        elif "compilationCheck_our" in strategy :
            
            compileRun = Compile_and_Run(self.target_lang, self.dataset_name, "inputOutput")
            for Data in tqdm(DataList):
                if Data["source_Lan"] in file_cont: continue
                if 'trans_code' not in Data or len(Data['trans_code'])==0: continue

                
                print("####",Data['source_Lan'])
                
                Method_signature = Data['target_method_signature'].strip()
                
                target_allMS = Method_signature
                if not Data['compileResult'] == 0 or Data['testResult']== "NOTEQUAL": 
                    with open(final_result_path, 'a', encoding='utf-8') as f:
                        json.dump(Data,f)
                        f.write('\n')    
                    continue
                
                Error_Numbers, ErrorMessageFunction, TestCase, ErrorMessage = compileRun.mainStart(Data, "compileRepair", isOur=self.isOur)
                
                error_number_list = []
                error_log = [""]
                iterative_count = 0
                invalidRepair_number = 0
                import_info = ""
                while Error_Numbers and invalidRepair_number<3: # set invalid repair number to 3
                    if ErrorMessage[-1]:
                        if self.target_lang == "cpp":ConvertExample = handcraftPrompt.MessageConver_parameterType_java.strip() + "\n\n" + handcraftPrompt.MessageConver_parameterType_cpp.strip()
                        else:ConvertExample = handcraftPrompt.MessageConver_parameterType_cpp.strip() + "\n\n" + handcraftPrompt.MessageConver_parameterType_java.strip()
                        Data['compileErrorType'] = "parameterType"
                        importInfo = ""
                        if self.model_type == 'chatglm2_6b':
                            ConvertExample = ""
                    else:
                        if self.target_lang == "cpp":ConvertExample = handcraftPrompt.MessageConver_not_parameterType_cpp.strip()
                        elif self.target_lang == "java": ConvertExample = handcraftPrompt.MessageConver_not_parameterType_java.strip()
                        elif self.target_lang == "python": ConvertExample = handcraftPrompt.MessageConver_not_parameterType_python.strip()
                        
                        if self.model_type == 'chatglm2_6b':
                            ConvertExample = ""
                        
                        Data['compileErrorType'] = "not_parameterType"
                        importInfo = "\n".join(set([code for code in Data['target_code_shell'].split("\n") if "import " in code or "#include" in code])) + "\n"
                    
                    errorConve_prompt = self.promptSelect.MessageConver(ErrorMessage[0].strip(), ErrorMessage[1].strip(), ErrorMessage[-1], ConvertExample, importInfo + ErrorMessage[2], self.target_lang)
                    print("errorConve_prompt:\n",errorConve_prompt)
                    Direct_ErrorMessage = ""
                    if self.model_type == "llama2_7B":
                        errorConve_response = self.model.generate(errorConve_prompt,False,True, False, 0, 128)
                        errorConve_response = errorConve_response.split("Error Location is as follows")[-1].split("Error Location is as follows")[-1]
                        Direct_ErrorMessage = "\n".join([line for line in errorConve_response.split("\n") if line.strip()][:3])
                        for errorInfo in errorConve_response.split("\n"):
                            if not errorInfo.strip(): continue
                            if Data['compileErrorType'] == "parameterType" and Data['target_method_signature'] in errorInfo: 
                                Direct_ErrorMessage  = errorInfo
                                break
                    else:      
                        errorConve_response = self.model.generate(errorConve_prompt,False,True, False, 0, 128)
                        print("errorConve_response:\n",errorConve_response)
                        
                        for errorInfo in errorConve_response.split("\n"):
                            if not errorInfo.strip(): continue
                            if "### " in errorInfo or errorInfo in Direct_ErrorMessage: break
                            Direct_ErrorMessage = Direct_ErrorMessage + "\n" +  errorInfo 
                    Data['errorConve_prompt'] = errorConve_prompt

                    print("Direct_ErrorMessage:\n",Direct_ErrorMessage)
                    
                    error_log.append(ErrorMessageFunction)
                    error_number_list.append(Error_Numbers)
                    if "<Buggy Line>" not in ErrorMessageFunction:
                        translated_code = ErrorMessageFunction
                        t_prompt = ""
                    else:
                        RepairExample = self.ExampleSelection(strategy)
                        if self.model_type == "chatglm2_6b":RepairExample = ""
                        t_prompt = self.promptSelect.CodeCompile_Repair_our(target_lang, ErrorMessageFunction.strip("\n"), target_allMS.strip(), TestCase.strip(), Direct_ErrorMessage, Data["commentTag"], RepairExample.strip())
                        if self.model_type == "chatglm2_6b": t_prompt = t_prompt.replace("\n### Example\n\n\n","")
                        print(t_prompt)
                        response = self.model.generate(t_prompt,False)
                        import_info, translated_code = self.responseDeal(response, target_lang, Method_signature)
                    if len(translated_code) == 0: break    
                    Data['trans_code'] = translated_code.strip()
                    Data['target_code_shell'] = import_info + "\n" + Data['target_code_shell']
                    Data['compileFix_prompt'] = t_prompt
                    Error_Numbers, ErrorMessageFunction, TestCase, ErrorMessage = compileRun.mainStart(Data, "compileRepair",isOur=self.isOur, boolStrategy=True)
                    if Error_Numbers >= error_number_list[-1]:
                        invalidRepair_number = invalidRepair_number + 1
                    if invalidRepair_number > 3 or Error_Numbers==0: break
                    if Error_Numbers ==  "TimeoutExpired": break
                    iterative_count = iterative_count + 1
                
                if iterative_count == 0: Data['prompt'] = "" 
                    
                if Error_Numbers == 0:
                    compileResult = 1

                    STATE, Group, blockNumber, InputOutput_function, _, _, _,_,_ = compileRun.mainStart(Data, "runRepair",boolStrategy=True)
                    if len(InputOutput_function)==0:
                        testResult = 1
                    elif InputOutput_function in ["NOTEQUAL","INST_ERROR"]: testResult = InputOutput_function
                    else: testResult = 0

                else:
                    compileResult = 0
                    testResult = 0
                    
                Data['compileResult'] =  compileResult
                Data['iterativeCount'] =  iterative_count
                Data['testResult'] = testResult
                Data['import_info'] = import_info
                with open(final_result_path, 'a', encoding='utf-8') as f:
                    json.dump(Data,f)
                    f.write('\n') 
                    
        elif "runCheckFSE" in strategy:
            compileRun = Compile_and_Run(self.target_lang, self.dataset_name, "compilationCheck")
            for Data in tqdm(DataList):
                iterativeCount = 0
                
                if Data["source_Lan"] in file_cont: continue
                print("Deal: ", Data['source_Lan'])
                if len(Data['all_input_output_value'].split("----")) < 1 or (self.target_lang != "python" and Data['compileResult'] ==0) or ("testResult" in Data and (Data['testResult'] == 1 or Data['testResult'] == "NOTEQUAL")):
                    with open(final_result_path, 'a', encoding='utf-8') as f:
                        json.dump(Data,f)
                        f.write('\n')    
                    continue
                
                Method_signature = Data['target_method_signature'].strip()
                STATE, Group, blockNumber, InputOutput_function, source_InstructCode, trans_InctructCode, source_message, trans_message, expectedActual = compileRun.mainStart(Data, "runRepair", boolStrategy=True)
                if len(expectedActual.split("--------")) <= 1 and STATE == "RunTimeERROR": 
                    with open(final_result_path, 'a', encoding='utf-8') as f:
                        json.dump(Data,f)
                        f.write('\n')    
                    continue
                
                if STATE == "RunTimeERROR": 
                    Example = self.ExampleSelection("runCheckFSE_RunTimerepair")
                    t_prompt = self.promptSelect.CodeRun_RunTimerepair(target_lang, Data['trans_code'].strip("\n"), Data['all_input_output_value'].split("----")[Group].strip("\n"),expectedActual.split("--------")[0].strip(),expectedActual.split("--------")[1].strip(),Method_signature, Example)
                elif STATE == "LogicERROR": 
                    Example = self.ExampleSelection("runCheckFSE_allIn")
                    t_prompt = self.promptSelect.CodeRun_AllInRunrepair(target_lang,  Data['trans_code'].strip("\n"), Data['all_input_output_value'].split("----")[Group].strip("\n"),expectedActual,Method_signature, Example.strip("\n"))
                else: 
                    if InputOutput_function == "NOTEQUAL": Data['testResult'] = "NOTEQUAL"
                    if InputOutput_function == "INST_ERROR": Data['testResult'] = "INST_ERROR"
                    Data['Noise'] = "Noise"
                    with open(final_result_path, 'a', encoding='utf-8') as f:
                        json.dump(Data,f)
                        f.write('\n')  
                    continue
                
                repair_log = [("NULL","NULL")]
                iterative_count_fixed = iterativeCount
                invalidRepair_number = 0
                import_info = ""
                try: 
                    while len(InputOutput_function) and iterative_count_fixed <= 3:
                        repair_log.append((Group, blockNumber))
                        
                        response = self.model.generate(t_prompt,False,True)
                        import_info, translated_code = self.responseDeal(response, target_lang, Method_signature)
                        if len(translated_code) == 0: break
                        Data['trans_code'] = translated_code
                        Data['target_code_shell'] = import_info + "\n" + Data['target_code_shell']
                        
                        Error_Numbers, ErrorMessageFunction, TestCase, ErrorMessage = compileRun.mainStart(Data, "compileRepair")
                        if Error_Numbers>0:
                            break
                        
                        STATE, Group, blockNumber, InputOutput_function, source_InstructCode, trans_InctructCode, source_message, trans_message, expectedActual = compileRun.mainStart(Data, "runRepair", boolStrategy=True)
                        iterative_count_fixed = iterative_count_fixed + 1
                        if len(expectedActual.split("--------")) <= 1 and STATE == "RunTimeERROR": break
                        if STATE == "RunTimeERROR":  
                            Example = self.ExampleSelection("runCheckFSE_RunTimerepair")
                            t_prompt = self.promptSelect.CodeRun_RunTimerepair(target_lang, Data['trans_code'].strip("\n"), Data['all_input_output_value'].split("----")[Group].strip("\n"),expectedActual.split("--------")[0].strip(),expectedActual.split("--------")[1].strip(),Method_signature, Example)
                        elif STATE == "LogicERROR": 
                            Example = self.ExampleSelection("runCheckFSE_allIn")
                            t_prompt = self.promptSelect.CodeRun_AllInRunrepair(target_lang,  Data['trans_code'].strip("\n"), Data['all_input_output_value'].split("----")[Group].strip("\n"),expectedActual,Method_signature, Example.strip("\n"))
                        else: break
                        
                    if len(InputOutput_function) == 0:
                        testResult = 1
                    else:
                        testResult = 0
                except Exception as e:
                        testResult = 0
                finally:
                    Data['run_iterativeCount'] =  iterative_count_fixed
                    Data['testResult'] = testResult
                    Data['import_info'] = import_info
                    Data['source_message'] = source_message
                    Data['trans_message'] = trans_message

                    with open(final_result_path, 'a', encoding='utf-8') as f:
                        json.dump(Data,f)
                        f.write('\n')  
                        
                         
        elif "runCheckTestFG" in strategy:  
            compileRun = Compile_and_Run(self.target_lang, self.dataset_name, "runCheckTestFG")
            source_commentTag = "#" if "python" in  DataList[0]['source_Lan'] else "//"
            
            for Data in tqdm(DataList):
                Data['iterativeCount'] = "NULL"
                InitialedCode = Data['trans_code']
                if Data["source_Lan"] in file_cont: continue
                print(Data['source_Lan'])
                

                
                if (self.target_lang != "python" and Data['compileResult'] ==0) or ("testResult" in Data and (Data['testResult'] == 1 or Data['testResult'] == "NOTEQUAL")):
                    with open(final_result_path, 'a', encoding='utf-8') as f:
                        json.dump(Data,f)
                        f.write('\n')    
                    continue
              
                repair_log = []
                iterative_count = 0
                invalidRepair_number = 0
                import_info = ""
                try: 
                    while True: # 开始进行修复
                        print("iterative_count: ", iterative_count)
                        

                        gen_cfg_path = os.path.join(current_dir, "gen_cfg",self.dataset_name,f"{self.source_lang}2{self.target_lang}")
                        self.boolean(gen_cfg_path)
                        codePath = self.fileWrite(Data, gen_cfg_path)  
                        BlockCode = BasicBlock.mainStart(codePath, Data['source_method_signature'].split(" ")[-1].strip())
                        BlockSourceCode = self.toolUse.functionExtraction(self.source_lang, BlockCode, Data['source_method_signature'].strip(), remove_comments=False)
                        
                        
                        codeMapping_Example = self.ExampleSelection("codeMapping")
                        t_prompt = self.promptSelect.CodeMapping(self.target_lang, Data['trans_code'], self.source_lang, BlockSourceCode.strip("\n"), codeMapping_Example.strip("\n"), source_commentTag)
                        response = self.model.generate(t_prompt,False)
                        Data['codeMapping'] = t_prompt + "\n----\n" + response
                        ts_BlockTransCode, ts_BlockSourceCode = self.responseCodeMapping(response, Data['trans_code'], BlockSourceCode)
                        
                        if ts_BlockSourceCode.count("BLOCK") != ts_BlockTransCode.count("BLOCK"): 
                            ts_BlockSourceCode, ts_BlockTransCode = self.BLOCKDeal(ts_BlockSourceCode, ts_BlockTransCode)
                        Data['source_code_block'] = ts_BlockSourceCode
                        Data['trans_code_block'] = ts_BlockTransCode
                        STATE, Group, blockNumber, InputOutput_function, _, _, _,trans_message,expectedActual = compileRun.mainStart(Data, "runRepair",isOur=True)
                        
                        if (Group, blockNumber) in repair_log: break
                        repair_log.append((Group, blockNumber))
                        if InputOutput_function == "NOTEQUAL": 
                            Data['testResult'] = "NOTEQUAL"
                            Data['iterativeCount'] =  iterative_count
                            break
                        elif STATE == "Others": 
                            Data['testResult'] = InputOutput_function
                            Data['iterativeCount'] =  iterative_count
                            break
                        
                        elif len(InputOutput_function) == 0:
                            Data['testResult'] = 1
                            Data['iterativeCount'] =  iterative_count
                            break
                        
                        if "Refine" in strategy:   
                            Example = ""   
                            genCode_prompt = self.promptSelect.TestFG_Refine(self.source_lang,self.target_lang, InputOutput_function[2].strip(), InputOutput_function[3].strip(),str(InputOutput_function[0]).strip(),InputOutput_function[1].strip(), Example)
                        elif "Further" in strategy:
                            Example = ""   
                            genCode_prompt = self.promptSelect.TestFG_Further(self.source_lang,self.target_lang, InputOutput_function[2].strip(), InputOutput_function[3].strip(),str(InputOutput_function[0]).strip(),InputOutput_function[1].strip(), Example) 
                        else:
                            Example = self.ExampleSelection("runCheckTestFG")
                            # Example = "\n"
                            genCode_prompt = self.promptSelect.TestFG(self.source_lang,self.target_lang, InputOutput_function[2].strip(), InputOutput_function[3].strip(),Example.strip())
                        Data['ExecuteFix_prompt'] = genCode_prompt
                        print("Inference: ....")    
                        response = self.model.generate(genCode_prompt, False)
                            
                        import_info, translated_code = self.responseDeal(response, self.target_lang, Data['target_method_signature'].strip())
                        if len(translated_code) == 0: break
                        Data['trans_code'] = translated_code.strip()
                        Data['target_code_shell'] = import_info + "\n" + Data['target_code_shell']
                        # 验证 修复的是否正确
                        Error_Numbers, ErrorMessageFunction, TestCase, ErrorMessage = compileRun.mainStart(Data, "compileRepair",  boolStrategy=True)
                        if Error_Numbers>0 and self.target_lang != "python": # 说明修坏了
                            break
                        
                        iterative_count = iterative_count + 1
                        STATE, Group, blockNumber, InputOutput_function, _, _, _,_,_ = compileRun.mainStart(Data, "runRepair", isOur=True, boolStrategy=True)
                        
                        if len(InputOutput_function) == 0:
                            Data['testResult'] = 1
                            Data['compileResult'] = 1
                            break  
                except Exception as e:
                    Data['Throw'] = "THROWERROR" 
                    
                finally:  
                    Data['iterativeCount'] =  iterative_count
                    with open(final_result_path, 'a', encoding='utf-8') as f:
                        json.dump(Data,f)
                        f.write('\n')                       
        
        elif "CodeAligner_alignment" in strategy:  
            source_commentTag = "#" if "python" in  DataList[0]['source_Lan'] else "//"
            
            final_result_path =f"./CodeTranslation/CodeTranslation/codeAlignment/Alignmented/{self.source_lang}2{self.target_lang}.jsonl"
            for Data in tqdm(DataList):
                print(Data['source_Lan'])

                if Data["source_Lan"] in file_cont: continue
                
                # gen_cfg_path = "./gen_cfg"
                gen_cfg_path = os.path.join(current_dir, "gen_cfg",self.dataset_name,f"{self.source_lang}2{self.target_lang}")
                self.boolean(gen_cfg_path)
                codePath = self.fileWrite(Data, gen_cfg_path)  
                BlockCode = BasicBlock.mainStart(codePath, Data['source_method_signature'].split(" ")[-1].strip())
                BlockSourceCode = self.toolUse.functionExtraction(self.source_lang, BlockCode, Data['source_method_signature'].strip(), remove_comments=False)
                
                
                codeMapping_Example = self.ExampleSelection("codeMapping")
                t_prompt = self.promptSelect.CodeMapping(self.target_lang, Data['trans_code'], self.source_lang, BlockSourceCode.strip("\n"), codeMapping_Example.strip("\n"), source_commentTag)
                response = self.model.generate(t_prompt,False)
                Data['CodeAligner_alignment'] = response
                with open(final_result_path, 'a', encoding='utf-8') as f:
                    json.dump(Data,f)
                    f.write('\n')                    
        
        elif "TransMap_alignment" in strategy:
            final_result_path =f"./CodeTranslation/CodeTranslation/codeAlignment/Alignmented/{self.source_lang}2{self.target_lang}.jsonl"
            source_commentTag = "#" if "python" in  DataList[0]['source_Lan'] else "//"
            tag = self.source_lang if self.source_lang != "python" else "py"
            
            for Data in tqdm(DataList):
                source_code_str = Data['source_code_str'].strip()
                source_code_list = source_code_str.split("\n")
                if Data["source_Lan"] in file_cont: continue
                
                stmt_source_code_list = []
                count = 1
                for code in source_code_list:
                    stmt = f" {source_commentTag} --- {tag} stmt {count}"
                    if not code.strip(): continue
                    new_code = code + stmt
                    stmt_source_code_list.append(new_code)
                    count = count + 1
                stmt_source_code_str = "\n".join(stmt_source_code_list)
                trans_code = Data['trans_code']
                codeMapping_Example = self.ExampleSelection("TransMap_alignment")
                t_prompt = self.promptSelect.TransMapping(self.source_lang, stmt_source_code_str.strip("\n"), self.target_lang, trans_code.strip("\n"), codeMapping_Example.strip("\n"))
                response = self.model.generate(t_prompt,False)
                Data['TransMap_alignment'] = response
                with open(final_result_path, 'a', encoding='utf-8') as f:
                    json.dump(Data,f)
                    f.write('\n')  
                
                
                
            
            codeMapping_Example = self.ExampleSelection("TransMap_alignment")         
    def BLOCKDeal(self, ts_BlockSourceCode, ts_BlockTransCode):
        source_BlockNumbers = [int(num) for num in re.findall(r'BLOCK(\d+)-START', ts_BlockSourceCode)]
        trans_BlockNumbers = [int(num) for num in re.findall(r'BLOCK(\d+)-START', ts_BlockTransCode)]
        source_BlockNumbers_set = set(source_BlockNumbers)
        trans_BlockNumbers_set = set(trans_BlockNumbers)
        Difference_List = list(source_BlockNumbers_set.difference(trans_BlockNumbers_set)) + list(trans_BlockNumbers_set.difference(source_BlockNumbers_set))
        
        source_commentTag = "#" if self.source_lang == "python" else "//"
        for value in Difference_List:
            start_block = f"{source_commentTag} BLOCK{value}-START"
            end_block = f"{source_commentTag} BLOCK{value}-END"
            ts_BlockSourceCode = ts_BlockSourceCode.replace(start_block,"")
            ts_BlockSourceCode = ts_BlockSourceCode.replace(end_block,"")
        
        trans_commentTag = "# " if self.target_lang == "python" else "// "
        for value in Difference_List:
            start_block = f"{trans_commentTag} BLOCK{value}-START"
            end_block = f"{trans_commentTag} BLOCK{value}-END"
            ts_BlockTransCode = ts_BlockTransCode.replace(start_block,"")
            ts_BlockTransCode = ts_BlockTransCode.replace(end_block,"")
        return ts_BlockSourceCode, ts_BlockTransCode
    
    def ExampleSelection(self, strategy):
        if strategy == "code2code":
            if "python" == self.source_lang: source_example = handcraftPrompt.python_trans_code_example
            elif "java" == self.source_lang: source_example = handcraftPrompt.java_trans_code_example
            elif "cpp" == self.source_lang: source_example = handcraftPrompt.cpp_trans_code_example
            
            if "python" == self.target_lang: target_example = handcraftPrompt.python_trans_code_example
            elif "java" == self.target_lang: target_example = handcraftPrompt.java_trans_code_example
            elif "cpp" == self.target_lang: target_example = handcraftPrompt.cpp_trans_code_example
            Method_example = "f_gold"
            return source_example, target_example, Method_example
        elif strategy ==  "TransMap_alignment":
            if "java" == self.source_lang and "python" == self.target_lang:
                Example = handcraftPrompt.transMap_java2python
            elif "java" == self.source_lang and "cpp" == self.target_lang:
                Example = handcraftPrompt.transMap_java2cpp
                
            elif "python" == self.source_lang and "cpp" == self.target_lang:
                Example = handcraftPrompt.transMap_python2cpp                
            elif "python" == self.source_lang and "java" == self.target_lang:
                Example = handcraftPrompt.transMap_python2java
                
            elif "cpp" == self.source_lang and "java" == self.target_lang:
                Example = handcraftPrompt.transMap_cpp2java
            elif "cpp" == self.source_lang and "python" == self.target_lang:
                Example = handcraftPrompt.transMap_cpp2python
            return Example
            
            
        elif strategy == "block_code2code":
            if "python" == self.source_lang: source_example = handcraftPrompt.python_block_code_example
            elif "java" == self.source_lang: source_example = handcraftPrompt.java_block_code_example
            elif "cpp" == self.source_lang: source_example = handcraftPrompt.cpp_block_code_example
            
            if "python" == self.target_lang: target_example = handcraftPrompt.python_block_code_example
            elif "java" == self.target_lang: target_example = handcraftPrompt.java_block_code_example
            elif "cpp" == self.target_lang: target_example = handcraftPrompt.cpp_block_code_example
            Method_signature_example = target_example.split("f_gold",1)[0] + "f_gold"
            return source_example, target_example, Method_signature_example
        elif strategy == "inputOutput":
            if "python" == self.source_lang: source_example = handcraftPrompt.python_trans_code_example
            elif "java" == self.source_lang: source_example = handcraftPrompt.java_trans_code_example
            elif "cpp" == self.source_lang: source_example = handcraftPrompt.cpp_trans_code_example
            
            if "python" == self.target_lang: target_example = handcraftPrompt.python_trans_code_example
            elif "java" == self.target_lang: target_example = handcraftPrompt.java_trans_code_example
            elif "cpp" == self.target_lang: target_example = handcraftPrompt.cpp_trans_code_example
            # Method_signature_example = target_example.split("f_gold",1)[0] + "f_gold"
            Method_signature_example = "f_gold"
            return source_example, target_example, Method_signature_example
        elif "compilationCheck_fse" in strategy:
            if "python" == self.target_lang:
                Example = handcraftPrompt.python_Compilerepair
            elif "java" == self.target_lang:
                Example = handcraftPrompt.java_Compilerepair
            elif "cpp" == self.target_lang:
                Example = handcraftPrompt.cpp_Compilerepair
            return Example
        elif "compilationCheck_our" in strategy:
            if "python" == self.target_lang:
                Example = handcraftPrompt.python_Compilerepair_our
            elif "java" == self.target_lang:
                Example = handcraftPrompt.java_Compilerepair_our
            elif "cpp" == self.target_lang:
                Example = handcraftPrompt.cpp_Compilerepair_our
            return Example
        elif "TypeInference" == strategy:
            if "python" == self.source_lang:
                Example = handcraftPrompt.typeInfer_python2java
            elif "java" == self.source_lang:
                Example  = handcraftPrompt.typeInfer_java2python
            return Example
        
        elif "runCheckFSE_allIn" in strategy:
            if "python" == self.target_lang:
                Example = handcraftPrompt.python_AllInRunrepair
            elif "java" == self.target_lang:
                Example = handcraftPrompt.java_AllInRunrepair
            elif "cpp" == self.target_lang:
                 Example = handcraftPrompt.cpp_AllInRunrepair  
            return Example
        elif "runCheckFSE_RunTimerepair" in strategy:
            if "python" == self.target_lang:
                Example = handcraftPrompt.python_RunTimerepair
            elif "java" == self.target_lang:
                Example = handcraftPrompt.java_RunTimerepair
            elif "cpp" == self.target_lang:
                Example = handcraftPrompt.cpp_RunTimerepair
            return Example
        elif "codeMapping" in strategy:
            if "java" == self.source_lang and "python" == self.target_lang:
                Example = handcraftPrompt.java2python_mapping
            elif "java" == self.source_lang and "cpp" == self.target_lang:
                Example = handcraftPrompt.java2cpp_mapping
                
            elif "python" == self.source_lang and "cpp" == self.target_lang:
                Example = handcraftPrompt.python2cpp_mapping                
            elif "python" == self.source_lang and "java" == self.target_lang:
                Example = handcraftPrompt.python2java_mapping
                
            elif "cpp" == self.source_lang and "java" == self.target_lang:
                Example = handcraftPrompt.cpp2java_mapping
            elif "cpp" == self.source_lang and "python" == self.target_lang:
                Example = handcraftPrompt.cpp2python_mapping
                
            return Example
        
        elif strategy == "FunctionGeration":
            if "python" == self.source_lang and "java" == self.target_lang:
                Example = handcraftPrompt.FG_python2java
            elif "python" == self.source_lang and "cpp" == self.target_lang:
                Example = handcraftPrompt.FG_python2cpp
            elif "java" == self.source_lang and "python" == self.target_lang:
                Example = handcraftPrompt.FG_java2python
            elif "java" == self.source_lang and "cpp" == self.target_lang:
                Example = handcraftPrompt.FG_java2cpp
            elif "cpp" == self.source_lang and "python" == self.target_lang:
                Example = handcraftPrompt.FG_cpp2python
            elif "cpp" == self.source_lang and "java" == self.target_lang:
                Example = handcraftPrompt.FG_cpp2java
                
            return Example
        
        elif strategy == "runCheckTestFG":
            if "java"== self.source_lang and "python" == self.target_lang:
                Example = handcraftPrompt.java2python_TestFG
            elif "java" == self.source_lang and "cpp" == self.target_lang:
                Example = handcraftPrompt.java2cpp_TestFG
            elif "python" == self.source_lang and "java" == self.target_lang:
                Example = handcraftPrompt.python2java_TestFG
            elif "python" == self.source_lang and "cpp" == self.target_lang:
                Example = handcraftPrompt.python2cpp_TestFG
                
            elif "cpp" == self.source_lang and "java" == self.target_lang:
                Example = handcraftPrompt.cpp2java_TestFG
            if "cpp"== self.source_lang and "python" == self.target_lang:
                Example = handcraftPrompt.cpp2python_TestFG
            return Example
        
        elif strategy == "Re_codeGeneration":
            if "python" == self.target_lang:
                Example = handcraftPrompt.RG_python
            elif "java" == self.target_lang:
                Example = handcraftPrompt.RG_java
            elif "cpp" == self.target_lang:
                Example = handcraftPrompt.RG_cpp
            return Example
        
        elif strategy == "FunctionGeration_MS":
            if "python" == self.source_lang and "java" == self.target_lang:
                Example = handcraftPrompt.FG_python2java_MS
            elif "java" == self.source_lang and "python" == self.target_lang:
                Example = handcraftPrompt.FG_java2python_MS
            return Example
        
        elif strategy == "Re_codeGeneration_MS":
            if "python" == self.target_lang:
                Example = handcraftPrompt.RG_python
            elif "java" == self.target_lang:
                Example = handcraftPrompt.RG_java_MS
            return Example

        elif "selfAnalyze" in strategy:
            if "python" == self.source_lang and "java" == self.target_lang:
                example = handcraftPrompt.python2java_TestFG_Analyze
            elif "python" == self.source_lang and "cpp" == self.target_lang:
                example = handcraftPrompt.python2cpp_TestFG_Analyze
            if "java" == self.source_lang and "python" == self.target_lang:
                example = handcraftPrompt.java2python_TestFG_Analyze
            if "java" == self.source_lang and "cpp" == self.target_lang:
                example = handcraftPrompt.java2cpp_TestFG_Analyze
            if "cpp" == self.source_lang and "java" == self.target_lang:
                example = handcraftPrompt.cpp2java_TestFG_Analyze
            if "cpp" == self.source_lang and "python" == self.target_lang:
                example = handcraftPrompt.cpp2python_TestFG_Analyze
            return example
                
        elif "runCheckTestFG_Refine" in strategy:
            if "python" == self.source_lang and "java" == self.target_lang:
                example = handcraftPrompt.python2java_TestFG_Refine
            elif "python" == self.source_lang and "cpp" == self.target_lang:
                example = handcraftPrompt.python2cpp_TestFG_Refine
            if "java" == self.source_lang and "python" == self.target_lang:
                example = handcraftPrompt.java2python_TestFG_Refine
            if "java" == self.source_lang and "cpp" == self.target_lang:
                example = handcraftPrompt.java2cpp_TestFG_Refine
            if "cpp" == self.source_lang and "java" == self.target_lang:
                example = handcraftPrompt.cpp2java_TestFG_Refine
            if "cpp" == self.source_lang and "python" == self.target_lang:
                example = handcraftPrompt.cpp2python_TestFG_Refine
            return example
            
        elif "runCheck" in strategy:
            if "python" == self.source_lang: source_example = handcraftPrompt.python_block_runRepair_correct
            elif "java" == self.source_lang: source_example = handcraftPrompt.java_block_runRepair_correct
            elif "cpp" == self.source_lang: source_example = handcraftPrompt.cpp_block_runRepair_correct
            
            if "python" == self.target_lang: 
                target_example = handcraftPrompt.python_block_runRepair_error
                repaired_target_example = handcraftPrompt.python_block_runRepair_correct
            elif "java" == self.target_lang: 
                target_example = handcraftPrompt.java_block_runRepair_error
                repaired_target_example = handcraftPrompt.java_block_runRepair_correct
            elif "cpp" == self.target_lang: 
                target_example = handcraftPrompt.cpp_block_runRepair_error
                repaired_target_example = handcraftPrompt.cpp_block_runRepair_correct
            return source_example, target_example, repaired_target_example
    
    def fileWrite(self, Data, Gen_outPath_dir):
        source_code = Data['source_code_str']
        if self.source_lang == "python":
            writenCode = Data['source_code_shell'].replace("# TOFILL", source_code)
        elif self.source_lang in ["java","cpp"]:
            writenCode =  Data['source_code_shell'].replace("// TOFILL", source_code)
        genefilePath = os.path.join(Gen_outPath_dir,Data['source_Lan'].split("###")[-1])
        with open(genefilePath, 'w', encoding='utf-8') as f:
            f.write(writenCode)
        return genefilePath
     
    def responseCodeMapping(self, response, source_code, blockCode):
        UnFind = []
        source_code = source_code + " "
        Source_commentTag = " # " if self.target_lang == "python" else " // "

        BlockSource_code = ""
        BLOCKList = re.split(r'\n(?=BLOCK\s*\d+:)', response)
        
        pattern = r"```(.*?)```"
        for block in BLOCKList:
            block_number_group = re.search(r'BLOCK\s*(\d+):', block)
            if not block_number_group: continue
            block_number = block_number_group.group(1)
            sourceCode_seg = re.findall(pattern, block, re.DOTALL)[-1] 
            sourceCode_seg = re.sub(r'(java|python|cpp)\n', '', sourceCode_seg, flags=re.IGNORECASE).strip()
            
            Start_Index, End_Index = self.find_original_index(source_code, sourceCode_seg)
            if Start_Index == -1: 
                UnFind.append((block_number, sourceCode_seg))
                continue
            ori_sourceCode_seg = source_code[Start_Index:End_Index]
            
            
            indentNumber = self.count_spaces_before_substring(source_code, ori_sourceCode_seg) 
            blockSourceCode = Source_commentTag + f" BLOCK{block_number}-START\n" + " "*indentNumber+ori_sourceCode_seg + "\n" + " "*indentNumber + Source_commentTag + f" BLOCK{block_number}-END\n"
            
            replaced_source_code = source_code[:End_Index].replace(ori_sourceCode_seg, blockSourceCode)
            BlockSource_code = BlockSource_code + replaced_source_code
            source_code = source_code[End_Index:]
            
        BlockSource_code = BlockSource_code + source_code
        BlockSource_code = "\n".join([code for code in BlockSource_code.split("\n") if code.strip()])
        
        pattern1 = re.compile(r'((?:#|//)\s*BLOCK\d+\s*-START.*?(?:#|//)\s*BLOCK\d+\s*-END)', re.DOTALL)
        for ori_block_number, sourceCode_seg in UnFind:
            blocks = pattern1.findall(BlockSource_code)
            for block in blocks:
                un_Indexs = [match.end() for match in re.finditer(re.escape(sourceCode_seg), block)]
                if len(un_Indexs): 
                    pattern = re.compile(r'BLOCK(\d+)-START')
                    block_numbers = pattern.findall(block)
                    target_block_number =  [int(num) for num in block_numbers][0]
                    blockCode = blockCode.replace("BLOCK"+str(ori_block_number), "BLOCK"+str(target_block_number)) 
        

        transformed_blockCode = []
        blockCodeLines = blockCode.split("\n")
        index = -1
        for i in range(len(blockCodeLines)):
            if i <= index or " ----" in blockCodeLines[i]: continue
            line1 = blockCodeLines[i]
            match1 = re.search(r'(#|//)\s*BLOCK(\d+)', line1)
            if match1:
                match1BlockNumber = match1.group(0)
                index = i
                for j in range(i+1, len(blockCodeLines)): 
                    line2 = blockCodeLines[j]
                    if re.search(r'(#|//)\s*BLOCK(\d+)', line2) and re.search(r'(#|//)\s*BLOCK(\d+)', line2).group(0) == match1BlockNumber:
                        index = j
                
                line = "\n".join([value.replace(match1BlockNumber,"") for value in blockCodeLines[i:index+1] if " ----" not in value])
                new_line = match1.group(0) + "-START" + "\n" +line + "\n" + match1.group(0) + "-END"  
                transformed_blockCode.append(new_line)
            else: transformed_blockCode.append(line1)
            
        return BlockSource_code, "\n".join(transformed_blockCode)    
    
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

            
    def responseFSDeal(self, response):
        dealFs = " ".join(response.replace("```","").split())
        
        return dealFs
    def responseTypeDeal(self, response):
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if not matches:
            inferedType = "\n".join([line for line in response.split("\n") if "After considering" not in line])
        else: inferedType = matches[0]
        return inferedType
    
    def responseVarUpdate(self, response):
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if not matches: return ""
        matchCode = re.sub(r'(python|java)\n', '', matches[0], flags=re.IGNORECASE)
        return matchCode
    
             
    def responseDeal(self, response, target_lang, Method_signature):
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL) 
        if not matches: 
            matches = [response]
        filterResponse = [match for match in matches if Method_signature.replace("_","").lower() in match.replace("_","").lower()]
        if len(filterResponse) == 0: return "",""
        for matchCode in filterResponse:
            matchCode = re.sub(r'(python|java|cpp)\n', '', matchCode, flags=re.IGNORECASE)
            import_info = "\n".join([code.strip().replace(". ",".").replace(" . ",".").replace(" .",".") for code in matchCode.split("\n") if "import " in code or "#include" in code])
            translated_code = self.toolUse.functionExtraction(target_lang, matchCode, Method_signature, remove_comments=True)
            if len(translated_code)>0:
                return import_info, translated_code
        return "",""

         
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

    # models = ['deepseek_7B']
    models = ['llama3_0_8B']
    # models = ['chatglm2_6b']
    
    import sys
    args = sys.argv[1:]
    tasks = []
    if len(args) >= 1:
        tasks.append(args[0])
    if len(args) >= 2:
        tasks.append(args[1])

    dataset_names = ['manually'] 
    # strategys = ['inputOutput','inputOutpuzt_compilationCheck_our','inputOutput_compilationCheck_fse', 'inputOutput_runCheckFSE', 'inputOutput_runCheckTestFG_Refine','inputOutput_runCheckTestFG_Further'] 
    strategys = ['inputOutput','inputOutput_compilationCheck_fse', 'inputOutput_runCheckFSE','inputOutput_compilationCheck_our', 'inputOutput_runCheckTestFG_Refine','inputOutput_runCheckTestFG_Further'] 
    # tasks = ["python2java","python2cpp","java2python","java2cpp"]   
    for model in models:
        for dataset_name in dataset_names:
            for strategy in strategys:
                for task in tasks:
                    print(f"model: {model}; dataset_name:{dataset_name}; strategy:{strategy}; task:{task}")
                    # print("task:", task)
                    source_lang = task.split("2")[0] 
                    target_lang = task.split("2")[-1]
                    output_path = os.path.join("./CodeTranslation/CodeTranslation",dataset_name,task+"-out")  
                    if "_our" in strategy: Main(source_lang, target_lang, model, strategy, dataset_name, True)
                    else: Main(source_lang, target_lang, model, strategy, dataset_name, False)