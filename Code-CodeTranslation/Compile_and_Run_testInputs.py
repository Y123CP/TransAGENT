import json
import os
import re
import shutil
import glob
from subprocess import Popen, PIPE
import subprocess
import pandas as pd
import shutil
from tqdm import tqdm
from definedTool import definedTool
from tree_sitter import Language, Parser
import ast
import math
import keyword
current_dir = os.path.dirname(__file__)
java_home = "./CodeTranslation/JavaEnv/jdk-17.0.11"


class Compile_and_Run:
    def __init__(self,target_lang, dataset_name, strategy):
        self.target_lang = target_lang
        self.dataset_name = dataset_name
        self.toolUse = definedTool()
        self.strategy = strategy
        self.genTemp = "./CodeTranslation/CodeTranslation/gen_temp"
        self.codeFuse_dirPath = "./CodeTranslation/ExperimentDataset/codefuseeval_cleaned"
        # self.boolStrategy = True if strategy in ['block_code2code', 'code2code', 'inputOutput','compilationCheck'] else False

    def mainStart(self, dataCont, Task, validTest=False, isOur=False, boolStrategy=False):
        self.boolStrategy = boolStrategy
        self.source_lang = dataCont['source_Lan'].split("###")[0]
        self.dataCont = dataCont
        self.task = Task
        self.isOur = isOur
        self.Method_signature = self.dataCont['target_method_signature']
            
        Gen_outPath_dir = os.path.join(self.genTemp, self.strategy, self.dataset_name, f"{self.source_lang}2{self.target_lang}")
        # self.target_lang = self.dataCont['target_Lan'].split("###")[0]

        if validTest == True:
            trans_functionCode = self.dataCont['reference_code'].strip()
            source_functionCode = self.dataCont['source_code_str'].strip()
        else:
            source_functionCode = self.dataCont['source_code_block'] if self.dataCont['source_code_block'] else self.dataCont['source_code_str']
            trans_functionCode = self.dataCont['trans_code_block'] if "trans_code_block" in self.dataCont and boolStrategy==False else self.dataCont['trans_code']
        
        # 1. Code Instrumention
        source_InstructCode = self.startInstruction(source_functionCode,self.source_lang)
        trans_InctructCode = self.startInstruction(trans_functionCode, self.target_lang)
        
        self.boolean(Gen_outPath_dir)
        source_dir = os.path.join(Gen_outPath_dir, self.source_lang)
        target_dir = os.path.join(Gen_outPath_dir, self.target_lang)
        self.boolean(source_dir)
        self.boolean(target_dir)
        
        source_codeShell = self.codeShellCollection(dataCont['source_code_shell'], self.source_lang, dataCont['source_method_signature'])
        trans_codeShell = self.codeShellCollection(dataCont['target_code_shell'], self.target_lang, dataCont['target_method_signature'])
        

        run_souce_code = self.addImport(source_codeShell.replace("$TOFILL$", source_InstructCode), self.source_lang)    
        source_genefilePath = os.path.join(source_dir, self.dataCont['source_Lan'].split("###")[-1])
        with open(source_genefilePath, 'w', encoding='utf-8') as f:
            f.write(run_souce_code)
        
        run_trans_code = self.addImport(trans_codeShell.replace("$TOFILL$", trans_InctructCode), self.target_lang)    
        target_genefilePath = os.path.join(target_dir, self.dataCont['target_Lan'].split("###")[-1])
        with open(target_genefilePath, 'w', encoding='utf-8') as f:
            f.write(run_trans_code)
        
        if self.task == "compileRepair":
            compileError = self.instruct_codeRun(target_genefilePath, self.target_lang) 
            if compileError == 0: return 0, "", "", ""
            if compileError == "TimeoutExpired": return "TimeoutExpired","TimeoutExpired","TimeoutExpired","TimeoutExpired"
            Error_Numbers, ErrorMessageFunction, TestCase, ErrorMessage = self.compilationError(compileError, target_genefilePath)
            return Error_Numbers, ErrorMessageFunction, TestCase, ErrorMessage
            
        elif self.task == "runRepair":
            source_message_temp, source_stderroutput = self.instruct_codeRun(source_genefilePath, self.source_lang)
            trans_message_temp, trans_stderroutput = self.instruct_codeRun(target_genefilePath, self.target_lang) 
            source_message, trans_message = self.printMessageFilter(source_message_temp), self.printMessageFilter(trans_message_temp)
            if len(source_stderroutput)>0 or len(source_message) == 0 or (len(trans_message)==0 and len(trans_stderroutput)==0):
                STATE = "Others"
                Group, blockNumber, expectedActual = "", "",  ""
                if  len(source_stderroutput)>0: InputOutput_function  = "INST_ERROR"
                else: InputOutput_function  = "ERROR"
                
            elif len(trans_stderroutput) == 0 or (len(trans_stderroutput) > 0 and self.isOur and self.boolStrategy == False):  # 判断是否存在逻辑错误/运行时错误
                if len(trans_stderroutput) > 0 and len(trans_message_temp) == 0:
                    Group, blockNumber, expectedActual, STATE = 0, "", "", "LogicERROR"
                    InputPrint = self.dataCont['all_input_output_value'].split("--------")[0].strip().split("\n")[0]
                    OutputPrint = "ERROR$$$$" + self.dataCont['all_input_output_value'].split("--------")[0].strip().split("\n")[-1]
                    allsourceBlockCode = self.dataCont['source_code_str']
                    transCodeRG_InputOutput = self.dataCont['trans_code'].strip().split("\n")[0] + "\n" + "[Fill in the Correct Code Logic Here]"
                    allTransBlockCode = self.dataCont['trans_code'].strip().split("\n")[0] + "\n" + "[Fill in the Correct Code Logic Here]"
                    InputOutput_function = [InputPrint, OutputPrint, allsourceBlockCode ,transCodeRG_InputOutput, allTransBlockCode]
                else:
                    runTimeError = False
                    if len(trans_stderroutput)>0: runTimeError = True
                    Group, blockNumber, InputOutput_function, expectedActual = self.blockValueCompare(source_message, trans_message, source_functionCode, trans_functionCode, trans_stderroutput, target_genefilePath, runTimeError)
                    
                    if len(InputOutput_function) > 0 and InputOutput_function not in ["NOTEQUAL"]: STATE = "LogicERROR"
                    elif len(InputOutput_function)==0: STATE = "Correct"
                    else: STATE = "Others"
                
            elif len(trans_stderroutput)>0 and trans_stderroutput != "TimeoutExpired": #  UniTrans used way
                STATE = "RunTimeERROR"
                blockNumber, InputOutput_function = "runError", "runError"
                Group  = trans_message.count("--------") 
                expectedActual = self.runTimeErrorRepair(trans_message, trans_stderroutput, target_genefilePath, trans_functionCode)

            else:
                STATE = "Others"
                Group, blockNumber,  InputOutput_function, expectedActual = "", "", "ERROR", ""
            
            return STATE, Group, blockNumber, InputOutput_function, source_InstructCode, trans_InctructCode, source_message, trans_message, expectedActual
    
    def printMessageFilter(self, Message):

        Message = Message.replace("CONDITION", '''{"CONDITION": "CONDITION"}''')
        block_message_list = Message.split("--------\n")

        Dealed_Message_list = []
        for block in block_message_list:
            BLOCK0TAG = 0
            if not block.strip(): continue
            Message_list = []
            for i in range(len(block.split("\n"))):
                mess = block.split("\n")[i]
                if "Block0:" in mess: BLOCK0TAG = BLOCK0TAG + 1
                if BLOCK0TAG >= 2:
                    if "RETURN" in Message_list[-1]:
                        blockNumber = re.findall("Block\d+:", Message_list[-1])[0]
                        blockValue = Message_list[-1].replace(blockNumber, "Expect_output:").replace("RETURN","FINAL")
                        Message_list.append(blockValue)
                    else:
                        FINAL_Value = block.split("\n")[i - 1]
                        Message_list.append(f''''Expect_output: {{"FINAL": {FINAL_Value}}}''')
                    break
                if "Expect_output" in mess or "Block" in mess:
                    Message_list.append(mess)
            Dealed_Message_list.append("\n".join(Message_list))
        return "\n--------\n".join(Dealed_Message_list)
        
    
              
    def addImport(self, codeCont, lang):
        if lang == "java":
            codeCont = "import com.google.gson.*;\nimport java.util.stream.Collectors;\nimport java.util.*;\n" + codeCont
        elif lang == "cpp":
            codeCont = "#include <nlohmann/json.hpp>\n" + codeCont
        elif lang == "python":
            codeCont = "import json\n" + codeCont
        return codeCont
    
    def compilationError(self, compile_info, genefilePath):
        file_name = os.path.basename(genefilePath)
        TestCase, ErrorMessage, ErrorCodeLine = "", "", ""
        
        if self.target_lang == "java":            
            compile_info_list = [info for info in compile_info.split("\n") if info.strip()]
            Error_Numbers = int(re.search(f"(\d+) error", compile_info).group(1))
            pattern = r"\.java:(\d+):" # Used for extract error_line from error_info
            
            # Extract Error_line, Error_Message
            for i in range(len(compile_info_list)):
                info = compile_info_list[i]
                if file_name in info and "error:" in info:
                    match = re.search(pattern, info)
                    Error_Line = int(match.group(1)) # type: ignore
                    Error_Message =  info.split("error:")[-1]
                        
                    if "cannot find symbol" in Error_Message: 
                        symbolInfo = ""
                        j = i+1
                        while True:
                            if j+1 == len(compile_info_list): break
                            if "symbol:" in compile_info_list[j]:
                                symbolInfo = compile_info_list[j].replace("symbol:","").strip()
                                break
                            j = j + 1
                        Error_Message = Error_Message + " `" + symbolInfo + " `"
                    break
            ErrorMessage = Error_Message 
                            
            i = 0
            allCode = []
            with open(genefilePath, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    i = i + 1
                    if int(Error_Line) == i:
                        indent_number = self.GetIndent(line)
                        errorLine = line.strip()
                        allCode.append(line.rstrip("\n") + " // <Buggy Line>"+ "\n")
                        ErrorCodeLine = line.rstrip("\n") 
                    else: allCode.append(line)
            ErrorMessageCode = " ".join(allCode)
            ErrorMessageFunction = self.toolUse.functionExtraction(self.target_lang, ErrorMessageCode, self.Method_signature, remove_comments=False)
            ErrorMessageFunction_list = [code for code in ErrorMessageFunction.split("\n") if code.strip()]
            ErrorMessageFunction = "\n".join(ErrorMessageFunction_list)
            ParameterError = False
            if "// <Buggy Line>" not in ErrorMessageFunction:
                ParameterError = True
                ErrorCodeLine = ErrorMessageFunction_list[0]
                ErrorMessageFunction_list[0] = ErrorMessageFunction_list[0] + " // <Buggy Line>"
                ErrorMessageFunction = "\n".join(ErrorMessageFunction_list)
                
            # TestCase = "".join(self.dataCont['input_output_value'])
            TestCase = self.dataCont['input_output_value'][0]
            
            if self.isOur:
                ErrorMessage = (f"Throw `{ErrorMessage}`, at `{errorLine}`", ErrorCodeLine.strip(), ErrorMessageFunction, ParameterError)
            return Error_Numbers, ErrorMessageFunction, TestCase, ErrorMessage
        elif self.target_lang == "cpp":
            compile_info_list = [info for info in compile_info.split("\n") if info.strip()]
            Error_Numbers = len([error_line for error_line in compile_info_list if file_name in error_line])
            
            pattern = r"\.cpp:(\d+)" # Used for extract error_line from error_info
            # Extract Error_line, Error_Message
            for i in range(len(compile_info_list)):
                info = compile_info_list[i]
                if file_name in info and re.search(pattern, info) and ("error:" in info or "error:" in compile_info_list[i+1]):
                    if len(compile_info_list) > i+1 and "error:" in compile_info_list[i+1]: info = compile_info_list[i] + " " + compile_info_list[i+1]
                    
                    Error_Line = int(re.search(pattern, info).group(1))
                    Error_Message = info.split("error:")[-1]
                    break       
            ErrorMessage =  Error_Message  
            i = 0
            allCode = []
            # add the error message into code
            with open(genefilePath, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    i = i + 1
                    if int(Error_Line) == i:
                        allCode.append(line.rstrip("\n") + " // <Buggy Line>"+ "\n")
                        ErrorCodeLine = line.rstrip("\n")
                        errorLine = line.strip()
                    else: allCode.append(line)
            ErrorMessageCode = " ".join(allCode)
            ErrorMessageFunction = self.toolUse.functionExtraction(self.target_lang, ErrorMessageCode, self.Method_signature, remove_comments=False)
            ErrorMessageFunction_list = [code for code in ErrorMessageFunction.split("\n") if code.strip()]
            ErrorMessageFunction = "\n".join(ErrorMessageFunction_list)
            ParameterError = False
            if "// <Buggy Line>" not in ErrorMessageFunction:
                ParameterError = True
                ErrorCodeLine = ErrorMessageFunction_list[0]
                
                ErrorMessageFunction_list[0] = ErrorMessageFunction_list[0] + " // <Buggy Line>"
                ErrorMessageFunction = "\n".join(ErrorMessageFunction_list)
            
            TestCase = self.dataCont['input_output_value'][0]
            
            if self.isOur:
                ErrorMessage = (f"Throw `{ErrorMessage}`, at `{errorLine}`", ErrorCodeLine.strip(), ErrorMessageFunction, ParameterError)
            return Error_Numbers, ErrorMessageFunction, TestCase, ErrorMessage
            
            
        elif self.target_lang == "python":
            Error_Numbers = 1  
            compile_info_list = [info for info in compile_info.split("\n") if info.strip()]
            
            Error_Line = 0
            pattern = r"line (\d+)"
            for info in reversed(compile_info_list):
                if file_name in info:
                    match = re.search(pattern,info)
                    Error_Line = int(match.group(1))
                    break
            
            ErrorMessage = compile_info_list[-1]

            if "expected an indented block" in ErrorMessage: indentTag = True  
            else: indentTag = False

            i = 0
            allCode = []
            # add the error message into code
            with open(genefilePath, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    i = i + 1
                    if int(Error_Line) == i:
                        indent_number = self.GetIndent(line)
                        errorLine = line.strip()
                        if indentTag: 
                            allCode.append("  "+line)
                        else: 
                            allCode.append(line.rstrip("\n") + " # <Buggy Line>" + "\n")
                        ErrorCodeLine = line.rstrip("\n")
                    else: 
                        allCode.append(line)
            ErrorMessageCode = " ".join(allCode)
            
            ErrorMessageFunction = self.toolUse.functionExtraction(self.target_lang, ErrorMessageCode, self.Method_signature, remove_comments=False)

            ErrorMessageFunction_list = [code for code in ErrorMessageFunction.split("\n") if code.strip()]
            ErrorMessageFunction = "\n".join(ErrorMessageFunction_list)
            if "# <Buggy Line>" not in ErrorMessageFunction:
                ErrorCodeLine = ErrorMessageFunction_list[0]
                ErrorMessageFunction_list[0] = ErrorMessageFunction_list[0] + " # <Buggy Line>"
                ErrorMessageFunction = "\n".join(ErrorMessageFunction_list)
            
            TestCase = self.dataCont['input_output_value'][0]
            if self.isOur:
                ErrorMessage = (f"Throw `{ErrorMessage}`, at `{errorLine}`", ErrorCodeLine.strip(), ErrorMessageFunction, False)
            return Error_Numbers, ErrorMessageFunction, TestCase, ErrorMessage
        
    def startInstruction(self, codeCont, lang):
        commentTag = " # " if lang == "python" else " // "
        self.deal_lang = lang
        PARSER_LOCATION = f"./CodeTranslation/JavaEnv/{self.deal_lang}_parser.so"
        LANGUAGE = Language(PARSER_LOCATION, self.deal_lang)
        self.parser = Parser()
        self.parser.set_language(LANGUAGE)
    
        self.codeCont = codeCont
        if self.boolStrategy: 
            runCode = codeCont
            if lang == "java": runCode = "\nprivate static HashMap<String, Object> InstruPrint = new HashMap<>();\n private static Gson gson = new GsonBuilder().disableHtmlEscaping().create();\n" + codeCont
            elif lang == "cpp": runCode = "using json = nlohmann::json;\njson InstruPrint;\n" + codeCont
            return runCode
        
        pattern1 = re.compile(r'((?:#|//)\s*BLOCK\d+\s*-START.*?(?:#|//)\s*BLOCK\d+\s*-END)', re.DOTALL)
        blocks = pattern1.findall(codeCont)
        self.blockedCode_list = [block.strip() for block in blocks]

        pattern2 = re.compile(r'(?:#|//)\s*BLOCK\d+\s*-START(.*?)(?:#|//)\s*BLOCK\d+\s*-END', re.DOTALL)
        instructCode = codeCont 
        self.StoredVariable = []
        self.InputOutput_Variable = []
        self.UnPrintVariable = []  
        self.python_setTypeVar = []
        self.allVariable = []  
        for i in range(len(self.blockedCode_list)):
            BLOCK_and_CODE = self.blockedCode_list[i]
            
            BlockNumber = [int(num) for num in re.findall(r'BLOCK(\d+)-START', BLOCK_and_CODE)][0]
            LineCode = pattern2.findall(self.blockedCode_list[i])[0]
            if lang in ["java","cpp"]: LineCode = self.remove_comments_java_cpp(LineCode)
            tree = self.parser.parse(bytes(LineCode, "utf8"))
            self.RemainingCode = codeCont.replace(LineCode,"")
            InstrumentLine = self.Instrumentation(LineCode, BlockNumber, tree.root_node)  
            BLOCK_and_InstrumentLine = BLOCK_and_CODE.replace(LineCode, InstrumentLine)
            instructCode = instructCode.replace(BLOCK_and_CODE, BLOCK_and_InstrumentLine)
            
        if self.deal_lang == "java": instructCode = "\nprivate static HashMap<String, Object> InstruPrint = new HashMap<>();\n private static Gson gson = new GsonBuilder().disableHtmlEscaping().create();\n" + instructCode
        if lang == "cpp": instructCode = "using json = nlohmann::json;\njson InstruPrint;\n" + instructCode
        return instructCode
    
    def codeShellCollection(self, CodeCont, lang, methodSig):
        methodSig = methodSig.split(" ")[-1]
        CodeCont_list = CodeCont.split("\n")
        placeholder = "$###$"
        TOFILLPlacehloder= "$TOFILL$"
        methodSig_stmts = []
        new_CodeCont_list = []
        for line in CodeCont_list:
            new_CodeCont_list.append(line)
            KeyIndex = self.indexFinder([methodSig],line)
            if len(KeyIndex) == 0: continue
            methodSig_param = self.BracketFinder(KeyIndex, line)
            methodSig_stmt = methodSig + methodSig_param
            indentNumber = self.GetIndent(line)
            methodSig_stmts.append(" "*indentNumber + methodSig_stmt)
            new_CodeCont_list.append(placeholder)
        
            
        new_CodeCont = "\n".join(new_CodeCont_list)
        if lang == "python":
            new_CodeCont = new_CodeCont.replace("# TOFILL",TOFILLPlacehloder)
            for methodSig_stmt in methodSig_stmts:
                indentPlaceHolder  = self.GetIndent(methodSig_stmt)
                methodSig_stmt = methodSig_stmt.strip()
                OutputExpression = " "* indentPlaceHolder + f"FINALOUTPUT = {methodSig_stmt}"
                FINAL1 = '''{json.dumps(list(k)) if isinstance(k, tuple) else k: v for k, v in FINALOUTPUT.items()}'''
                printStmtIF = '''{"FINAL":''' + FINAL1 +'''}'''
                printStmtElse = '''{"FINAL": FINALOUTPUT}'''
                IFStmt = " "* indentPlaceHolder + f'''if isinstance(FINALOUTPUT, dict): print(\"Expect_output:\",json.dumps({printStmtIF}));'''
                ElseStmt = " "* indentPlaceHolder + f'''else: print(\"Expect_output:\",json.dumps({printStmtElse}));'''
                allPrintStmt = OutputExpression +"\n"+f"{IFStmt}\n{ElseStmt}" + "\n" + " "* indentPlaceHolder + "print(\"--------\")"
                new_CodeCont = new_CodeCont.replace(placeholder, allPrintStmt, 1)
                    
        elif lang == "java":
            new_CodeCont = new_CodeCont.replace("// TOFILL",TOFILLPlacehloder)
            for methodSig_stmt in methodSig_stmts:               
                if len([code for code in new_CodeCont_list if methodSig_stmt in code and "HashMap" in code and not code.split("HashMap",1)[0].strip()]) > 0:
                    allPrintStmt = f"InstruPrint.put(\"FINAL\", {methodSig_stmt}.entrySet().stream().filter(e -> e.getValue() != null).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue))); System.out.println(\"Expect_output:\"+gson.toJson(InstruPrint)); System.out.println(\"--------\");InstruPrint.clear();"+ "\n"
                else:
                    allPrintStmt = f"InstruPrint.put(\"FINAL\",{methodSig_stmt}); System.out.println(\"Expect_output:\"+gson.toJson(InstruPrint)); System.out.println(\"--------\");InstruPrint.clear();"+ "\n" 
                new_CodeCont = new_CodeCont.replace(placeholder, allPrintStmt, 1)
                
        elif lang == "cpp":
            new_CodeCont = new_CodeCont.replace("// TOFILL",TOFILLPlacehloder)
            for methodSig_stmt in methodSig_stmts:
                allPrintStmt = f'''InstruPrint["FINAL"] = {methodSig_stmt}; std::cout << "Expect_output: " << InstruPrint.dump() << std::endl; std::cout << "--------" << std::endl; InstruPrint.clear();'''
                new_CodeCont = new_CodeCont.replace(placeholder, allPrintStmt, 1)
        return new_CodeCont
        
              
    def replace_expression(self, var,str1):
        pattern = rf'{var}\s*=\s*[^;]*;'
        result = re.sub(pattern, '#$$$#', str1)
        return result
    
    def remove_repeatValue(self, message_list):
        new_message_list = []
        for message in message_list:
            message_0 = message.split("\n")[0]
            message_1 = message.split(message_0)[-1]
            new_message_list.append(message_0+"\n"+message_1)
        return new_message_list
    
    
    def blockValueCompare(self,source_message, trans_message, source_functionCode, trans_functionCode, trans_stderroutput, target_genefilePath, runTimeError=False) -> str:
        source_message_list = self.remove_repeatValue([item.lstrip() for item in source_message.split("--------") if item.strip()])
        trans_message_list = self.remove_repeatValue([item.lstrip()  for item in trans_message.split("--------") if item.strip()])
        if len(source_message_list) != len(trans_message_list) and runTimeError==False: return "","","NOTEQUAL",""
        def discrepancyValue():  
            diff_index = -1
            block_number = 0
            for group in range(len(source_message_list)):  
                sub_source_message_list = [item for item in source_message_list[group].split("\n") if item.strip() and ("Block" in item or "Expect_output" in item)]
                sub_trans_message_list = [item for item in trans_message_list[group].split("\n") if item.strip() and ("Block" in item or "Expect_output" in item)]
                
                if (len(sub_source_message_list) == 0 or len(sub_trans_message_list)==0) and runTimeError == False: return -1, -1, -1, -1
                elif "Expect_output:" in sub_trans_message_list[-1]:
                    source_output = json.loads(sub_source_message_list[-1].split("Expect_output:")[-1])
                    trans_output = json.loads(sub_trans_message_list[-1].split("Expect_output:")[-1])
                    if "FINAL" in source_output and "FINAL" in trans_output and self.compare_values(source_output["FINAL"], trans_output["FINAL"]): continue  
                    elif (len(source_output) == 0 or len(trans_output) == 0) and (("FINAL" in source_output and source_output['FINAL']==None) or ("FINAL" in trans_output and trans_output['FINAL']==None)): continue
                    elif ("FINAL" not in source_output) and ("FINAL" in trans_output and len(trans_output)==0): continue
                    elif ("FINAL" not in trans_output) and ("FINAL" in source_output and len(source_output)==0): continue
                    elif self.boolStrategy: 
                        expectedOutput = source_output['FINAL'] if "FINAL" in source_output else ""
                        actuallyOutput = trans_output['FINAL'] if "FINAL" in trans_output else ""
                        expectedActual = f"Expected Output: {expectedOutput}, Acutal Output: {actuallyOutput}"
                        return group,"ERROR","",expectedActual
                
                for j in range(len(sub_source_message_list)-1):  
                    if j >= len(sub_trans_message_list) or "Expect_output" in sub_trans_message_list[j]: 
                        if len(trans_stderroutput) != 0:  
                            block_number = self.runTimeErrorRepair(trans_message, trans_stderroutput, target_genefilePath, trans_functionCode, iterateLast=True).split("###")[-1].lower().capitalize()+":"
                            j_temp = [j for j in range(len(sub_source_message_list)-1) if block_number in sub_source_message_list[j]]
                            ts_j = j_temp[0] if len(j_temp)!=0 else "NULL_J"
                            
                        elif len(trans_stderroutput) == 0: 
                            ts_j = j-1
                            block_number = re.findall("Block\d+:", sub_source_message_list[j-1])[0]
                        
                        return group, ts_j, [block_number, block_number], "" 
                    
                    source_blockValue = sub_source_message_list[j]
                    trans_blockValue = sub_trans_message_list[j]
                    
                    block_number_source = re.findall("Block\d+:", source_blockValue)[0]  
                    block_number_trans = re.findall("Block\d+:", trans_blockValue)[0]
                            
                    if block_number_source != block_number_trans:  
                        if "CONDITION" in sub_source_message_list[j-1]:
                            conditionBlock = sub_source_message_list[j-1]  
                            conditionBlock_number = re.findall("Block\d+:", conditionBlock)[0]
                            return group, j, [block_number_source, block_number_trans, conditionBlock_number, True], ""
                        else:
                            No_conditionBlock = sub_trans_message_list[j] 
                            No_conditionBlock_number = re.findall("Block\d+:", No_conditionBlock)[0]
                            return group, j, [block_number_source, block_number_trans, No_conditionBlock_number, False], ""


                    else:
                        block_number  = block_number_source
                        source_values =  json.loads(re.split("Block\d+:",source_blockValue)[-1])
                        trans_values =  json.loads(re.split("Block\d+:",trans_blockValue)[-1])
                        

                        if len(trans_values) == 0: 
                            return group, j, [block_number, block_number], ""
                        shared_keys = self.find_common_keys(source_values, trans_values)
                        for key1,key2 in shared_keys:
                            if not self.compare_values(source_values[key1], trans_values[key2]) and "Block0" in block_number_trans:
                                return "","NOTEQUAL","","" 
                            elif not self.compare_values(source_values[key1], trans_values[key2]) and "Block0" not in block_number_trans:
                                return group, j, [block_number, block_number], ""
                        if j+1 == len(sub_source_message_list)-1:
                            return group, -1, [0, block_number],""
                       
            return 0, diff_index, [0, block_number], ""
            
        if len(trans_message) != 0:  
            group, diff_index, diff_blockNumber, expectedActual = discrepancyValue()
            if diff_index == "ERROR": return group, "", "ERROR", expectedActual        
            if diff_index == -1: return "", "", "","" 
            if diff_index == "NOTEQUAL": return "","","NOTEQUAL",""
            
            if diff_index == "NULL_J":
                runTimeInfo = self.runTimeErrorRepair(trans_message, trans_stderroutput, target_genefilePath, trans_functionCode, iterateLast=True)
                diff_blockNumber = [runTimeInfo.split("###")[-1].lower().capitalize()+":", runTimeInfo.split("###")[-1].lower().capitalize()+":"]
                Error_Message = runTimeInfo.split("###")[0]
                expectedActual = Error_Message
                Input_dict, expect_output_dict = "",""
            else:
                sub_trans_message_list = [item for item in trans_message_list[group].split("\n") if item.strip() and ("Block" in item or "Expect_output" in item)]
                sub_source_message_list = [item for item in source_message_list[group].split("\n") if item.strip() and ("Block" in item or "Expect_output" in item)]
                Input_dict, expect_output_dict = self.Input_and_Output(sub_source_message_list, diff_index, diff_blockNumber, functionCode = source_functionCode)
                if len(trans_stderroutput) == 0 or trans_stderroutput == 'TimeoutExpired':
                    Actually_input_dict, Actually_output_dict = self.Input_and_Output(sub_trans_message_list, diff_index, diff_blockNumber, functionCode = trans_functionCode)
                    if type(Actually_input_dict) == dict:
                        for key, value in Actually_input_dict.items():
                            if key not in Input_dict:
                                Input_dict[key] = value
                    else: Actually_output_dict = trans_stderroutput
                            
                else: 
                    errorLine_errorInfo = self.runTimeErrorRepair(trans_message, trans_stderroutput, target_genefilePath, trans_functionCode)
                    Actually_output_dict = "{"+errorLine_errorInfo.split("--------")[1] + "} at BUGGY LINE" 
                    errorLine = errorLine_errorInfo.split("--------")[0]
                    transCommentTag = "# " if self.target_lang == "python" else "// "
                    trans_functionCode = trans_functionCode.replace(errorLine, errorLine.rstrip('\n') + f" {transCommentTag}BUGGY LINE\n")
                expectedActual = expect_output_dict
            
        elif len(trans_message) == 0:  
            if self.boolStrategy: return "","","ERROR",""
            runTimeInfo = self.runTimeErrorRepair(trans_message, trans_stderroutput, target_genefilePath, trans_functionCode,iterateLast=True)
            diff_blockNumber = [runTimeInfo.split("###")[-1].lower().capitalize()+":", runTimeInfo.split("###")[-1].lower().capitalize()+":"]
            Error_Message = runTimeInfo.split("###")[0]
            group, expectedActual = 0, Error_Message
            Input_dict, expect_output_dict = "",""

        if diff_blockNumber[0] == diff_blockNumber[1]: 
            repairBlock = diff_blockNumber[0]
            ActuallyOutput = Actually_output_dict
        else: 
            repairBlock = diff_blockNumber[2]
            if diff_blockNumber[3]:
                if int(diff_blockNumber[0].replace("Block","").replace(":","")) < int(diff_blockNumber[1].replace("Block","").replace(":","")):
                    ActuallyOutput = "The Condition is Return FALSE"
                    expect_output_dict = "The Condition is Return TRUE"
                else:
                    ActuallyOutput = "The Condition is Return TRUE"
                    expect_output_dict = "The Condition is Return FALSE"
            else:
                ActuallyOutput = Actually_output_dict
        sourceCommentTag = "# " if self.source_lang == "python" else "// "
        sourceBlockCode = []
        allsourceBlockCode = []
        sourceCodeList = source_functionCode.split("\n")
        index = -1
        for i in range(len(sourceCodeList)):
            # if index != -1: break
            if i < index: continue
            line = sourceCodeList[i]
            blockNumber = re.findall("BLOCK\d+",line)
            if blockNumber and blockNumber[0].lower() == repairBlock.replace(":","").lower() and "START" in line:
                sourceBlockCode.append(f"{sourceCommentTag} ------1.Input is {Input_dict}------")
                allsourceBlockCode.append(f"{sourceCommentTag} ------1------")
                index = i
                while True:
                    index = index + 1
                    blockNumber2 = re.findall("BLOCK\d+",sourceCodeList[index])
                    if blockNumber2 and blockNumber2[0].lower() == repairBlock.replace(":","").lower() and "END" in sourceCodeList[index]:
                        sourceBlockCode.append(f"{sourceCommentTag} ------2.Output is {expect_output_dict}------")
                        allsourceBlockCode.append(f"{sourceCommentTag} ------2------")
                        break
                    elif blockNumber2: continue
                    else: 
                        sourceBlockCode.append(sourceCodeList[index])
                        allsourceBlockCode.append(sourceCodeList[index])
            elif blockNumber: continue  # BLOCK -START/-END 
            else: allsourceBlockCode.append(line)

        allTransBlockCode = []
        transCodeRG_InputOutput = []
        transCodeList = trans_functionCode.split("\n")
        transCommentTag = "# " if self.target_lang == "python" else "// "
        index = -1
        InputPrint, OutputPrint = "",""
        for i in range(len(transCodeList)):
            if i < index: continue
            line = transCodeList[i]
            blockNumber = re.findall("BLOCK\d+",line)
            if blockNumber and blockNumber[0].lower() == repairBlock.replace(":","").lower() and "START" in line:
                
                InputPrint = Input_dict
                transCodeRG_InputOutput.append(f"{transCommentTag} ------1------")
                allTransBlockCode.append(f"{transCommentTag} ------1------")
                
                index = i
                while True:
                    index = index + 1
                    blockNumber2 = re.findall("BLOCK\d+",transCodeList[index])
                    if blockNumber2 and blockNumber2[0].lower() == repairBlock.replace(":","").lower() and "END" in transCodeList[index]:
                        transCodeRG_InputOutput.append(f"{transCommentTag} ------2------")
                        allTransBlockCode.append(f"{transCommentTag} ------2------")
                        OutputPrint = f"{ActuallyOutput} $$$${expect_output_dict}"
                        break
                    else: 
                        allTransBlockCode.append(transCodeList[index])
                        FIM = " "*self.GetIndent(transCodeList[index]) + "[Fill in the Correct Code Here!]"
                        if FIM.strip() in "\n".join(transCodeRG_InputOutput): continue
                        else: transCodeRG_InputOutput.append(FIM)
                        
            elif blockNumber: continue  # BLOCK -START/-END 
            else: 
                transCodeRG_InputOutput.append(line)
                allTransBlockCode.append(line)
            # sourceBlockCode
        return group, diff_blockNumber[0], [InputPrint, OutputPrint, "\n".join(allsourceBlockCode) ,"\n".join(transCodeRG_InputOutput), "\n".join(allTransBlockCode)], expectedActual
        
    
    
    def compare_values(self, value1, value2, precision=0.005):        
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            # If both are numeric, compare with specified precision
            return abs(value1 - value2) < precision or math.isinf(abs(value1 - value2))
        elif isinstance(value1, dict) and isinstance(value2, dict):
            # If both are dictionaries, compare keys and values
            if set(value1.keys()) != set(value2.keys()):
                return False
            return all(self.compare_values(value1[k], value2[k]) for k in value1)
        elif isinstance(value1, list) and isinstance(value2, list):
            # If both are lists, compare each element
            return len(value1) == len(value2) and all(self.compare_values(a, b) for a, b in zip(value1, value2))
        elif isinstance(value1, str) and "[" in value1 and "]" in value1 and "[" in value2 and "]" in value2 and (".0" in value1 or ".0" in value2):
            return value1.replace(".0","") == value2.replace(".0","")
        elif not isinstance(value1, (int, float)) and not isinstance(value2, (int, float)): #  and len(value1) == 0 and len(value2) == 0
            return True
        else:
            # For all other types, compare directly
            return value1 == value2
        

    def dict_to_key_equals_value(self, d):
        items = []
        for key, value in d.items():
            if isinstance(value, str):
                items.append(f"{key}='{value}'")  
            else:
                items.append(f"{key}={value}")  
        return '{' + ', '.join(items) + '}'

    def find_common_keys(self, dict1, dict2): 
        def normalize_key(key):
            return key.replace('_', '').lower()
        normalized_dict1 = {normalize_key(k): k for k in dict1}
        normalized_dict2 = {normalize_key(k): k for k in dict2}
        
        common_keys = set(normalized_dict1.keys()) & set(normalized_dict2.keys())
        common_key_pairs = [(normalized_dict1[k], normalized_dict2[k]) for k in common_keys]
        
        return common_key_pairs
)
    def Input_and_Output(self, sub_message_list, diff_index, diff_blockNumber, functionCode):
        if len(sub_message_list) == 1: return "", ""
        
        start_block_number = diff_blockNumber[0]
        end_block_number = diff_blockNumber[1]
        
        StoredVariable = []
        StoredDict = []
        block_list = []

        for n in range(len(sub_message_list)):
            message = sub_message_list[n]
            block_number = re.findall("Block\d+:", message)[0]  
            message_noBlockNumber = json.loads(re.split("Block\d+:",message)[-1])
            # message_noBlockNumber = self.preprocess_dict_keys(json.loads(re.split("Block\d+:",message)[-1]))
            if n == diff_index and (block_number == start_block_number or block_number == end_block_number): 
                Input_var = set(message_noBlockNumber)
                

                output_dict = message_noBlockNumber
                output_dict = self.reorder_dict_by_function_code(output_dict, functionCode, block_number)
                
                # statement is for statement
                ForTag = False
                functionCode_list = functionCode.replace(" (","(").split("\n")
                for var in range(len(functionCode_list)):
                    code = functionCode_list[var]
                    BLOCK = block_number.replace(":","").upper()
                    if f"{BLOCK}-" in code and "-END" not in code:
                        python_ForCondition = re.search("for\s+\w+\s+in ",functionCode_list[var+1]) and ":" in functionCode_list[var+1] and "-END" in functionCode_list[var+2]
                        javaC_ForCondition = "for(" in functionCode_list[var+1] and (functionCode_list[var+1].count("{")!=functionCode_list[var+1].count("}"))
                        if python_ForCondition or javaC_ForCondition: 
                            ForTag = True
                            break
                if ForTag: 
                    output_list = []
                    allOutput_dict = [block.replace(block_number,"") for block in sub_message_list if block_number in block]
                    allKeys = output_dict.keys()
                    for key in allKeys:
                        output_list.append([key, [json.loads(value_dict)[key] for value_dict in allOutput_dict]])
                    output_list2 = []
                    for key, value in output_list:
                        output_list2.append(f"`{key}` iterates through the range is {value}")
                    output_dict = "; ".join(output_list2)
                        
                Input_dict = {}
                for var in Input_var:
                    for dictItem in reversed(StoredDict):
                        if var in dictItem:
                            Input_dict[var] = dictItem[var]
                            break

                if len(Input_dict) == 0: Input_dict = message_noBlockNumber
                Input_dict = self.reorder_dict_by_function_code(Input_dict, functionCode, block_number)
                

                return Input_dict, output_dict
            StoredVariable.extend(list(message_noBlockNumber.keys()))
            StoredDict.append(message_noBlockNumber)
            block_list.append(block_number)

    def reorder_dict_by_function_code(self, input_dict, function_code, BLOCK_NUMBER):
        BLOCK_NUMBER = BLOCK_NUMBER.upper().replace(":","").strip()
        Start_Block = BLOCK_NUMBER+"-START"
        End_Block = BLOCK_NUMBER + "-END"
        BlockCode = function_code.split(Start_Block)[-1].split(End_Block)[0]
        
        keys = list(input_dict.keys())
        if len(keys) > 1:
            if "RETURN" in keys: keys.remove("RETURN")
            key_positions = {key: min(match.start() for match in re.finditer(rf'{re.escape(key)}', BlockCode)) for key in keys}
            sorted_keys = sorted(key_positions, key=key_positions.get, reverse=True)
        
            sorted_dict = {key: input_dict[key] for key in sorted_keys}
            if "RETURN" in input_dict: sorted_dict['RETURN'] = input_dict['RETURN']
        else: sorted_dict = input_dict
        return sorted_dict

    def runTimeErrorRepair(self, trans_message, stderroutput, target_genefilePath, trans_functionCode, iterateLast=False):
        errorMessage_list = stderroutput.split("\n")
        if self.target_lang == "python": 
            error_info = [line for line in reversed(errorMessage_list) if "File " in line and ", line" in line][0]
            Error_Line = int(re.search(r'line (\d+)', error_info).group(1))
            Error_Message = [info for info in errorMessage_list if info.strip()][-1]
            with open(target_genefilePath, 'r', encoding='utf-8') as f:
                CODE = f.read()
            
            i = 0
            allCode = []
            ErrorLine = ""
            with open(target_genefilePath, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    i = i + 1
                    if int(Error_Line) == i:
                        ErrorLine = line + "--------"+Error_Message
                        if iterateLast: 
                            errorBlock = self.errorBlock(line.strip(), CODE.replace(" (","("))
                            ErrorLine = Error_Message + "###" + errorBlock
                        break
                    else: 
                        allCode.append(line)
            return ErrorLine
        
        elif self.target_lang == "java": 
            fileName = os.path.basename(target_genefilePath)
            error_info = [line for line in reversed(errorMessage_list) if fileName in line][-1]
            Error_Line = int(re.search(r'.java:(\d+)', error_info).group(1))
            Error_Message = [info for info in errorMessage_list if info.strip()][0]
            with open(target_genefilePath, 'r', encoding='utf-8') as f:
                CODE = f.read()
            i = 0
            allCode = []
            ErrorLine = ""
            with open(target_genefilePath, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    i = i + 1
                    if int(Error_Line) == i:
                        ErrorLine = line + "--------"+Error_Message
                        if iterateLast: 
                            errorBlock = self.errorBlock(line.strip(), CODE.replace(" (","("))
                            ErrorLine = Error_Message + "###" + errorBlock
                        break
                    else: 
                        allCode.append(line)
            return ErrorLine
        
        elif self.target_lang == "cpp":
            fileName = os.path.basename(target_genefilePath)
            with open(target_genefilePath, 'r', encoding='utf-8') as f:
                CODE = f.read()
            with open(target_genefilePath, 'r') as file:
                lines = file.readlines()
            if stderroutput == "TimeoutExpired": 
                if re.findall("BLOCK\d+",CODE):
                    errorBlock = re.findall("BLOCK\d+",CODE)[-1]
                else: errorBlock = ""
                return "TimeoutExpired###"+errorBlock
        
            Error_Message, line = "", ""
            for index in range(len(errorMessage_list)):
                errorInfo = errorMessage_list[index]
                if "Program received" in errorInfo:
                    Error_Message = errorInfo
                elif fileName in errorInfo and "main()" not in errorInfo and "In function" not in errorInfo:
                    Error_Line = int(re.search(r'.cpp:(\d+)', errorInfo).group(1))
                    line = lines[Error_Line - 1].strip()
                    ErrorLine = line + "--------"+Error_Message
                    if iterateLast: 
                        errorBlock = self.errorBlock(line.split("        ")[-1].strip(), CODE.replace(" (","("))
                        ErrorLine = Error_Message + "###" + errorBlock
                    break
                    
                    
            return ErrorLine


    def errorBlock(self,errorLine, trans_functionCode):
        errorBlock = ""
        startTag = False
        for code in trans_functionCode.split("\n"):
            if errorLine in code:
                startTag = True
            if startTag:
                if "BLOCK" in code:
                    re.findall("BLOCK\d+",code)
                    return re.findall("BLOCK\d+",code)[0]
        return errorBlock
                       
    def extract_return_expression(self, code_str, lang):
        if code_str.strip().endswith(";"): code_str = code_str.replace(";","")
        code_str = [code for code in code_str.split("\n") if code.strip() and code.strip()!="}"][-1]
        pattern = r"return(.*)"
        match = re.search(pattern, code_str)
        if match: 
            expression = match.groups()[-1]
            if lang == "java": expression = expression.replace(";","")
            return expression  
        else: return None  
    
    def starts_with_return(self, code_str):
        if not re.search(rf'\b{re.escape("return")}\b', code_str): return False
        cleaned_str = [code for code in code_str.split("\n") if code.strip() and code.strip() != "}"][-1]
        return_split = cleaned_str.split("return")
        return not bool(re.search('[a-zA-Z]', return_split[0]))


    def indexFinder(self, keywords, sourceStr):
        pattern = r'\b(' + '|'.join(keywords) + r')\b|\b(' + '|'.join(keywords) + r')(?=\W)'
        matchKeys = re.finditer(pattern, sourceStr)
        KeyIndex = [(match.group(1), match.start(1), match.end(1)) for match in matchKeys]
        return KeyIndex

    def BracketFinder(self,  KeyIndex, Code, leftChar="(", rightChar=")"):
        T_code = self.preprocess_string(Code)
        
        Condition_list = []
        stack = []
        for index in KeyIndex:
            End = index[2]
            if T_code[End] != leftChar: continue
            for i in range(End, len(T_code)):
                char = T_code[i]
                if char == leftChar:
                    stack.append(i)
                elif char == rightChar and stack:
                    start = stack.pop()
                    if not stack:
                        Condition_list.append(" ".join(Code[start: i+1].split()))
                        break
        return "\n".join(Condition_list)
            
    
    def instruct_codeRun(self,file_path, lang):
        if lang == "python":
            if self.task == "compileRepair":
                try:
                    run_result = subprocess.run(f"python {file_path}", shell=True, capture_output=True, text=True, timeout=10)
                    if "FINAL" in run_result.stderr and "Object of type set is not JSON serializable" in run_result.stderr:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            pythonCode = f.read()
                        new_pythonCode = pythonCode.replace('''"FINAL":''', '''"FINAL":list(''').replace('''}));''', ''')}));''')
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_pythonCode)
                        run_result = subprocess.run(f"python {file_path}", shell=True, capture_output=True, text=True, timeout=10)
                    stdoutput = run_result.stdout
                    stderroutput = run_result.stderr
                    if len(stderroutput) == 0: return 0  
                    else: 
                        if "incompatible types" in stderroutput: return 0  
                        else: return stderroutput
                except subprocess.TimeoutExpired as e:
                    return 0
            elif self.task == "runRepair":
                try:
                    run_result = subprocess.run(f"python {file_path}", shell=True, capture_output=True, text=True, timeout=10)
                    if "FINAL" in run_result.stderr and "Object of type set is not JSON serializable" in run_result.stderr:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            pythonCode = f.read()
                        new_pythonCode = pythonCode.replace('''"FINAL":''', '''"FINAL":list(''').replace('''}));''', ''')}));''')
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_pythonCode)
                        run_result = subprocess.run(f"python {file_path}", shell=True, capture_output=True, text=True, timeout=15)
                    stdoutput = run_result.stdout
                    stderroutput = run_result.stderr
                    return stdoutput, stderroutput
                except subprocess.TimeoutExpired as e:
                    output_str = e.stdout.decode('utf-8') if e.stdout else ""
                    return output_str[:20000], "TimeoutExpired"
        elif lang == "java":
            javaFile = file_path
            
            javafx_path = "./CodeTranslation/JavaEnv/javafx-sdk-17.0.9/lib/*"
            googlJson_path = "./CodeTranslation/JavaEnv/gson-2.8.9.jar"
            javac_command = f"{java_home}/bin/javac -cp {javafx_path}:{googlJson_path}:{file_path} {javaFile}"

            if self.task == "compileRepair":
                try:
                    return_result =subprocess.run(javac_command, capture_output=True, shell=True, timeout=25)
                    if return_result.returncode == 0: return 0
                    else: return return_result.stderr.decode('utf-8')
                except subprocess.TimeoutExpired as e:
                    return "TimeoutExpired"
                
            
            elif self.task == "runRepair":
                try:
                    return_result = subprocess.run(javac_command, capture_output=True, shell=True, timeout=25)
                    if return_result.returncode != 0: return return_result.stdout.decode('utf-8'), return_result.stderr.decode('utf-8')
                    os.chdir(os.path.dirname(file_path))
                    class_name = os.path.splitext(os.path.basename(file_path))[0]
                    java_command = f"{java_home}/bin/java -cp {javafx_path}:{googlJson_path}:{os.path.dirname(file_path)} {class_name}"
                    run_result = subprocess.run(java_command, shell=True, capture_output=True, text=True, timeout=10)
                    os.chdir(current_dir)
                    stdoutput = run_result.stdout
                    stderroutput = run_result.stderr
                    return stdoutput, stderroutput
                except subprocess.TimeoutExpired as e:
                    os.chdir(current_dir)
                    output_str = e.stdout.decode('utf-8') if e.stdout else ""
                    return output_str[:20000], "TimeoutExpired"
        elif lang == "cpp":
            targetCodePath = file_path
            nlohmannDir = "./CodeTranslation/JavaEnv/cpp_nlohmann/json/include"
            output_executable =  os.path.splitext(os.path.basename(targetCodePath))[0]
            directory = os.path.dirname(targetCodePath)
            compile_command = ["g++", "-Wall", "-g","-I", nlohmannDir, targetCodePath, "-o", os.path.join(directory, output_executable)]
            if self.task == "compileRepair":
                compile_result = subprocess.run(compile_command, capture_output=True, text=True,errors='replace')
                if compile_result.returncode == 0: return 0
                else: return compile_result.stderr
            elif self.task == "runRepair":
                try:
                    compile_result = subprocess.run(compile_command, capture_output=True, text=True,errors='replace')
                    if compile_result.returncode != 0: return compile_result.stdout, compile_result.stderr
                    gdb_command = ["gdb", "--batch", "-ex", "run", "-ex", "bt", os.path.join(directory, output_executable)]
                    run_result = subprocess.run(gdb_command, capture_output=True, text=True, timeout=20, errors='replace')
                    if "No stack." in run_result.stderr: 
                        return run_result.stdout, ""
                    else: 
                        return run_result.stdout, run_result.stdout

                except subprocess.TimeoutExpired as e:
                    output_str = e.stdout if e.stdout else ""
                    if type(output_str)!=str: output_str = output_str.decode('utf-8') 
                    return output_str[:20000], "TimeoutExpired"
    

    def reBlockTrans(self, trans_blockCode)->str:
        trans_block_list = trans_blockCode.split("\n")
        
        find_number = 0  
        blockTag = False
        new_block = [trans_block_list[0]]
        sub_block = []
        for i in range(len(trans_block_list)): 
            line = trans_block_list[i]
            if "-----" in line and blockTag == False:  
                start_number = ''.join(filter(str.isdigit, line))
                sub_block.append(line)
                blockTag = True
            elif "-----" not in line and blockTag == True: 
                sub_block.append(line)
            
            elif "-----" in line and blockTag == True: 
                code_element = "".join("\n".join(sub_block).split())
                Only_braces = all(char == '}' for char in code_element)
                if Only_braces: 
                    continue
                else: 
                    start_number = str(int(start_number) + 1)
                    new_line = "// ----------" + start_number + "----------"
                    sub_block.append(new_line)
                    new_block.extend(sub_block)
                    sub_block.clear()
            if i+1 == len(trans_block_list):
                start_number = str(int(start_number) + 1)
                new_line = "// ----------" + start_number + "----------"

                last_index = -1  
                for index, item in enumerate(sub_block):
                    if isinstance(item, str) and re.search(r'[a-zA-Z]', item):
                        last_index = index
                
                if last_index != -1:
                    sub_block.insert(last_index + 1, new_line)
                
                new_block.extend(sub_block)
                sub_block.clear()
                
        return "\n".join(new_block)


    def GetIndent(self, s):
        return len(s) - len(s.lstrip(' '))  
    
    def boolean(self,file_path):
        if not os.path.exists(file_path):
            print('Creat floder....')
            os.makedirs(file_path)
        else:
            shutil.rmtree(file_path)
            os.makedirs(file_path)


    def FormatCode(self, filepath, lang):
        if lang == "cpp":
            command = ["clang-format", "-i", filepath]
            subprocess.run(command)
        elif lang == "java":
            java_home = "./CodeTranslation/JavaEnv/jdk-11.0.21/bin/java"
            google_java_path = "./CodeTranslation/JavaEnv/google-java-format-1.21.0-all-deps.jar"
            java_command = [java_home, "-jar", google_java_path, "--replace", filepath]
            subprocess.run(java_command)
        elif lang == "python":
            subprocess.run(["./.conda/envs/zqc_py38/bin/black", filepath])


    def Instrumentation(self, code, i, Node):
        if self.splitMethod(code): code = code.replace(" (", "(")
        codeCont = self.codeCont.replace(" (", "(")
        variables = list(set(self.extract_variables(Node))) 
        bodyRemovedCode = self.removeBody(code)
        
        if self.deal_lang == "java":
            if "switch(" in code or "case " in code or "throw new" in code: return code
            
            out_code = ""
            Inner_code = ""
            Inner_codeList = code.split(" -> ", 1)
            if len(Inner_codeList) >= 2 and "((" in code:
                Inner_code = code.split("((", 1)[-1]
                out_code = code.split("((", 1)[0]
            elif "for(" not in code or "@Override" in code:
                Inner_codeList = [code for code in code.split(") {", 1) if code.strip()]
                if len(Inner_codeList) >= 2 and "}" in code:
                    pattern = r'\{([^\{\}]*?(?:\{[^\{\}]*?\}[^\{\}]*?)*?)\}'
                    Inner_code = re.findall(pattern, code)
                    out_code = code.replace(Inner_code[0],"") if len(Inner_code) else Inner_codeList[0]
                    Inner_code = Inner_code[0] if len(Inner_code)>=1 else Inner_code
            elif "for(" in code and "{" in code and code.split("for(",1)[-1].count("{") == code.split("for(",1)[-1].count("}"):
                pattern = re.compile(r'for\s*\(.*?\)\s*{.*?}', re.DOTALL)
                matches = pattern.findall(code)
                out_code = code
                Inner_code = ""
                for match in matches:
                    Inner_code = Inner_code + "\n" + match 
                    out_code = out_code.replace(match,"")
            elif "for(" in code and "{" not in code:
                out_code = code.split("for(")[0]
                Inner_code = code.split("for(")[1]
                
            valuePrint = []
            for var in variables:  
                if var in self.allVariable and len(var.strip())>1:  
                    valuePrint.append(var)
                    continue
                pattern1 = r"(?<=\(|\s|\[)" + re.escape(var) + r"(?![a-zA-Z<_\-({\.:0-9])" 
                pattern2 = r"(?<=\(|\s|\[)" + re.escape(var) + r"\."  
                pattern3 = rf"\.{re.escape(var)}\b"
                if ((re.search(pattern1, bodyRemovedCode) or (re.search(pattern2, code) and re.search(pattern1, codeCont))) and not re.search(pattern3, codeCont)) and var.strip() not in ["Integer", "String", "gson","InstruPrint", "else","None","BigInteger"]:
                    if ((not self.IsVariable_initialized(bodyRemovedCode, var, code) or
                            re.search(rf"> {var};", code) or
                            re.search(rf"{var} ->", code)) or
                            (re.search(rf'\b{re.escape(var)}\b', Inner_code) and not re.search(rf'\b{re.escape(var)}\b', out_code))): continue 
                    elif len(re.findall(rf"\b{var}\b", self.RemainingCode))>=1:
                        valuePrint.append(var)

            if self.IfWhile(bodyRemovedCode):
                KeyIndex = self.IfWhile(bodyRemovedCode, True)
                condition_list_t = self.conditionExtraction_java_cpp(KeyIndex, bodyRemovedCode)
                condition_list = [code for code in condition_list_t if code not in Inner_code]
                valuePrint.extend(condition_list)
                
            if self.starts_with_return(bodyRemovedCode):
                return_condition = self.extract_return_expression(bodyRemovedCode, "java")
                withoutputReturn = code.split(return_condition)[0]
                for var in valuePrint:
                    if not re.search(rf"\b{var}\b", withoutputReturn):
                        valuePrint.remove(var)
                if return_condition and not return_condition.strip().lstrip('-').isdigit(): valuePrint.append(return_condition+"$RETURN$")

            if len(valuePrint) > 0 and "else return" not in code:
                valuePrint = list(set(valuePrint))
                escaped_vars = []
                for var in valuePrint:
                    if "$RETURN$" not in var:
                        escaped_vars.append((var.replace('"', '\\"'), var))
                    else:
                        escaped_vars.append(("RETURN", var.replace("$RETURN$","")))
                
                variables_string = ' '.join(f"InstruPrint.put(\"{name}\",{value});" for name, value in escaped_vars)
                print_statement = f'''{variables_string} System.out.println("Block{i}:"+gson.toJson(InstruPrint)); InstruPrint.clear();\n'''
                
                if self.starts_with_return(bodyRemovedCode):
                    code_list = code.split("\n")
                    return_index= [i for i in range(len(code_list)) if "return" in code_list[i]][0]
                    if return_index>0 and re.search('[a-zA-Z]', code_list[0]):
                        while return_index-1 >=0 and not re.search('[a-zA-Z]', code_list[return_index-1]):
                            return_index = return_index - 1
                        
                    code_list[return_index] = " " * self.GetIndent(code) + print_statement + "\n" + code_list[return_index]
                    InstruCode = "\n".join(code_list)
                elif code.count("{") == code.count("}"): 
                    if (self.RemainingCode.split(f"BLOCK{i}-END")[-1].strip().startswith('else') or 
                        self.RemainingCode.split(f"BLOCK{i+1}-START")[-1].strip().startswith('else')) and code.strip().endswith("}"):
                        InstruCode = code[:code.rfind('}')] + "\n" + print_statement + code[code.rfind('}'):] 
                    else: InstruCode = code + "\n" + print_statement
                
                else:
                    code_list = code.split("\n")
                    letter_index = [i for i in range(len(code_list)) if bool(re.search('[a-zA-Z]', code_list[i]))][-1]
                    code_list[letter_index] = code_list[letter_index]  +  "\n" + print_statement 
                    InstruCode = "\n".join(code_list)
                    
            elif len(variables)>0 and not any(item in ["Integer", "String", "gson","InstruPrint", "else"] for item in variables):
                print_statement = f'''System.out.println("Block{i}:"+gson.toJson(InstruPrint)); InstruPrint.clear();'''
                InstruCode = "\n" + " "*self.GetIndent(code) + print_statement + "\n" + code
            else: 
                InstruCode = code
            if ("if(" in code or "while(" in code) and (code.count("{")!= code.count("}") and "{" in code) and "else if" not in code:
                print_statement =  "\n" + f'''System.out.println("Block{i}: "+"CONDITION");'''
                InstruCode = print_statement + "\n" + InstruCode
                
            self.allVariable.extend(valuePrint)
            return InstruCode

        elif self.deal_lang == "python":
            out_code = self.removeBody(code)
            if "for " in code and "=" in code: 
                out_code = re.split(r'\s*=\s*\[', code, 1)[0]
            inner_code = ""
            if "for " not in code and code.count("(") == code.count(")"):
                inner_code = " ".join(self.extract_parentheses_content(code))

            valuePrint = []
            for var in variables:
                pattern1 = r"(?<=\(|\s|\[)" + re.escape(var) + r"(?![a-zA-Z<_\-({\.0-9])" 
                pattern2 = r"(?<=\(|\s|\[)" + re.escape(var) + r"\."  
                if re.search(f'{var}\s*=\s*set\(', codeCont) or re.search(rf"{var}\s*=\s*SortedList", codeCont): 
                    valuePrint.append(f"__builtins__.list({var})")
                elif f"lambda {var}:" in out_code or re.search(rf"{var}\s*=\s*lambda", out_code) or (f'''for {var} in ''' in out_code and not out_code.strip().endswith(":")):
                    continue
                elif not len(re.findall(rf"\b{var}\b", self.RemainingCode))>=1 and (re.search(pattern1, inner_code) or re.search(pattern2, inner_code)):  # 如果某个变量在括号当中，同时没有出现在其他地方，这个var 不能print
                    continue
                elif (re.search(pattern1, out_code) or (re.search(pattern2, out_code) and re.search(pattern1, codeCont))) and var.strip() not in ["elif", "except", "else", "True","reverse","break","int","List","Tuple","Set","bool",'str', 'ord','float','default']:
                    valuePrint.append(var)
                elif var in self.allVariable:
                    valuePrint.append(var)
                    
            if self.IfWhile(out_code):
                KeyIndex = self.IfWhile(out_code, True)
                condition_list = self.conditionExtraction_python(KeyIndex, out_code)
                for cond in condition_list:
                    if "re.search" in cond:
                        valuePrint.append("str(" + cond + ")")
                    else:
                        valuePrint.append(cond)
            if self.starts_with_return(code):
                # valuePrint.clear()
                return_condition = self.extract_return_expression(code, "python")
                if len(return_condition): # and return_condition.strip() not in self.allVariable
                    if f"{return_condition.strip()} = set(" in codeCont or f"{return_condition.strip()} = {{" in codeCont:
                        return_condition = f"__builtins__.list({return_condition})"
                    elif "," in return_condition and ("(" not in return_condition or "{" not in return_condition):
                        return_condition = f"({return_condition})"
            
                    ## remove the return_condition from the code
                    withoutputReturn = code.split(return_condition)[0]
                    for var in valuePrint:
                        if not re.search(rf"\b{var}\b", withoutputReturn):
                            valuePrint.remove(var)
                    valuePrint.append(return_condition+"$RETURN$")
                

            if len(valuePrint) > 0 and "else return" not in code:
                valuePrint = list(set(valuePrint))
                escaped_vars = []
                for var in valuePrint:
                    if "$RETURN$" not in var:
                        escaped_vars.append((var.replace('"', '\\"'), var))
                    else:
                        escaped_vars.append(("RETURN", var.replace("$RETURN$","")))
                        
                var_list = [(re.sub(r'__builtins__\.list\(([^)]+)\)', r'\1', name), value) for name, value in escaped_vars]
                
                variables_string = '''{''' + ",".join(f"\"{name}\":{value}" for name, value in var_list) + "}"
                print_statement = f'''print("Block{i}:", json.dumps({variables_string}))'''                
                
                ts_code = [value for value in code.split("\n") if value.strip()][-1]
                if "if " in code and "else:" in code and "return " not in code:
                    InstruCode = code.rstrip()+ "\n" +  " " * self.GetIndent(code.lstrip('\n'))  + print_statement + "\n"
                elif self.pythonCodeType(ts_code.strip()):
                    pattern = r'[a-zA-Z0-9=]'
                    if ("while " in code or "if " in code) and bool(re.search(pattern, code.split(":")[-1])):
                        InstruCode = code.rstrip()+ "\n" +  " " * self.GetIndent(code.lstrip('\n'))  + print_statement + "\n"
                    else: InstruCode = code.rstrip() + "\n" + " " * self.GetIndent(ts_code.lstrip('\n')) + "    "  + print_statement + "\n"
                
                elif self.starts_with_return(ts_code):
                    ReturnInstruCode = "\n" + " " * self.GetIndent(ts_code.lstrip('\n'))  + print_statement + "\n" + ts_code.rstrip()+ "\n"
                    InstruCode = code.replace(ts_code, ReturnInstruCode)
                else:
                    InstruCode = code.rstrip()+ "\n" +  " " * self.GetIndent(ts_code.lstrip('\n'))  + print_statement + "\n"
            else:
                InstruCode = code
            
            
            if ("if " in code or "while " in code) and (":" in code) and "elif " not in code:
                ifWhile_code = [value for value in code.split("\n") if value.strip() and ("if " in code or "while " in code)][0]
                ifWhile_codeIndent = self.GetIndent(ifWhile_code.lstrip('\n'))
                lastCode = [value for value in code.split("\n") if value.strip()]
                lastCode_codeIndent = self.GetIndent(lastCode[-1].lstrip('\n'))
                if lastCode_codeIndent > ifWhile_codeIndent or len(lastCode) == 1:  
                    print_statement = f'''print("Block{i}: ", "CONDITION")'''
                    ts_code = [value for value in code.split("\n") if value.strip()][0]
                    InstruCode = "\n" + " " * self.GetIndent(ts_code.lstrip('\n'))  + print_statement + "\n" + InstruCode
                    
            self.allVariable.extend(valuePrint)
            return InstruCode

        elif self.deal_lang == "cpp":
             
            out_code = ""
            Inner_code = ""
            Inner_codeList = code.split(" -> ", 1)
            if len(Inner_codeList) >= 2:
                Inner_code = code.split("((", 1)[-1]
                out_code = code.split("((", 1)[0]
            elif "for(" not in code or "@Override" in code:
                Inner_codeList = [code for code in code.split(") {", 1) if code.strip()]
                if len(Inner_codeList) >= 2 and "}" in code:
                    pattern = r'\{([^\{\}]*?(?:\{[^\{\}]*?\}[^\{\}]*?)*?)\}'
                    Inner_code = re.findall(pattern, code)
                    out_code = code.replace(Inner_code[0],"") if len(Inner_code) else Inner_codeList[0]
                    Inner_code = Inner_code[0] if len(Inner_code)>=1 else Inner_code

            elif "for(" in code and "{" in code and code.split("for(",1)[-1].count("{") == code.split("for(",1)[-1].count("}"):
                pattern = re.compile(r'for\s*\(.*?\)\s*{.*?}', re.DOTALL)
                matches = pattern.findall(code)
                out_code = code
                Inner_code = ""
                for match in matches:
                    Inner_code = Inner_code + "\n" + match 
                    out_code = out_code.replace(match,"")
            elif "for(" in code and "{" not in code:
                out_code = code.split("for(")[0]
                Inner_code = code.split("for(")[1]
                
            valuePrint = []
            for var in variables:  
                if var in self.allVariable: 
                    valuePrint.append(var)
                    continue
                pattern1 = r"(?<=\(|\s|\[|&)" + re.escape(var) + r"(?![a-zA-Z<_\-({\.:0-9])" 
                pattern2 = r"(?<=\(|\s|\[)" + re.escape(var) + r"\."  
                pattern3 = rf"\.{re.escape(var)}\b"
                if ((re.search(pattern1, bodyRemovedCode) or (re.search(pattern2, code) and re.search(pattern1, codeCont))) and not re.search(pattern3, codeCont)) and var.strip() not in ["auto", "String", "gson","InstruPrint", "else","None"]:
                    if ((not self.IsVariable_initialized(bodyRemovedCode, var, code) or
                            re.search(rf"> {var};", code) or
                            re.search(rf"{var} ->", code)) or
                            (re.search(rf'\b{re.escape(var)}\b', Inner_code) and not re.search(rf'\b{re.escape(var)}\b', out_code))): continue # 此 var 未初始化
                    elif re.search(rf"\b{re.escape(var)}\s*=",code) and "{\n" in code.split(var)[-1] and not len(re.findall(rf"\b{var}\b", self.RemainingCode))>=1: 
                        self.UnPrintVariable.append(var)
                        continue 
                    elif len(re.findall(rf"\b{var}\b", self.RemainingCode))>=1 and var not in self.UnPrintVariable:
                        valuePrint.append(var)

            if self.IfWhile(bodyRemovedCode):
                KeyIndex = self.IfWhile(bodyRemovedCode, True)
                condition_list_t = self.conditionExtraction_java_cpp(KeyIndex, bodyRemovedCode)
                condition_list = [code for code in condition_list_t if code not in Inner_code]
                valuePrint.extend(condition_list)
                
            if self.starts_with_return(bodyRemovedCode):
                return_condition = self.extract_return_expression(bodyRemovedCode, "java")
                withoutputReturn = code.split(return_condition)[0]
                for var in valuePrint:
                    if not re.search(rf"\b{var}\b", withoutputReturn):
                        valuePrint.remove(var)
                if return_condition and not return_condition.strip().lstrip('-').isdigit(): valuePrint.append(return_condition+"$RETURN$")

            if len(valuePrint) > 0 and "else return" not in code:
                valuePrint = list(set(valuePrint))
                escaped_vars = []
                for var in valuePrint:
                    if "$RETURN$" not in var:
                        escaped_vars.append((var.replace('"', '\\"'), var))
                    else:
                        escaped_vars.append(("RETURN", var.replace("$RETURN$","")))

                variable_list = []
                for name, value in escaped_vars:
                    # 
                    pattern = rf'\b{re.escape(name)}\[.*?\]'  # 数组
                    
                    pattern2 = rf"stack<.*>\s*?\b{re.escape(name)}\s*[;,]"
                    pattern3 = rf'''stack<.*>\s*\b\w+\b\s*,\s*{re.escape(name)}\b\s*;'''
                    
                    pattern4 = rf"priority_queue<.*>\s*?\b{re.escape(name)}\s*[;,]"
                    pattern5 = rf'(std::)?vector\s*<\s*(std::)?bitset<\d+>\s*>\s+\b{re.escape(name)}\b'
                    
                    if re.search(rf"string {re.escape(name)}",codeCont) and re.search(pattern, codeCont): 
                        variable_list.append(f"InstruPrint[\"{name}\"]={value};")
                    elif re.search(pattern5, codeCont):
                        InstruPrint = '''InstruPrint = {{"$NAME$", [&$NAME$]() { std::vector<std::string> vec; for (const auto& bs : $NAME$) vec.push_back(bs.to_string()); return vec; }()}};'''
                        InstruPrint = InstruPrint.replace("$NAME$",name)
                        variable_list.append(InstruPrint)
                        
                    elif re.search(pattern, codeCont) and self.dataset_name == "manually":
                        if re.search(rf"unordered_map<.*>.*?\b{re.escape(name)}", codeCont) or re.search(rf"map<.*>.*?\b{re.escape(name)}", codeCont):
                            InstruPrint_map = '''for (const auto& [key, value] : $NAME$) { InstruPrint["$NAME$"][std::to_string(key)] = value;} '''
                            InstruPrint_map = InstruPrint_map.replace("$NAME$",name)
                            variable_list.append(InstruPrint_map)
                        else:
                            if re.search(rf"vector<.*>.*?\b{re.escape(name)}", codeCont): size = f"{name}.size()"
                            elif name == "d": size = "n+1"
                            else: size = "size"
                            variable_list.append(f'''InstruPrint["{name}"] = json::array(); for (int j = 0; j < {size}; ++j) InstruPrint["{name}"].push_back({name}[j]); ''')
                    elif re.search(pattern2, codeCont) or re.search(pattern3, codeCont):
                        InstruPrint = '''InstruPrint = {{"$NAME$", [&$NAME$] { std::vector<int> vec; while (!$NAME$.empty()) { vec.push_back($NAME$.top()); $NAME$.pop(); } return vec; }()}};'''
                        InstruPrint = InstruPrint.replace("$NAME$",name)
                        variable_list.append(InstruPrint)
                    elif re.search(pattern4, codeCont):
                        InstruPrint = '''InstruPrint = {{"$NAME$", [&$NAME$] { std::vector<int> vec; while (!$NAME$.empty()) { vec.push_back($NAME$.top()); $NAME$.pop(); } return vec; }()}};'''
                        InstruPrint = InstruPrint.replace("$NAME$",name)
                        variable_list.append(InstruPrint)
                        
                    elif re.search(rf'auto {re.escape(name)}',codeCont) and not re.search(rf'auto {re.escape(name)}\s*:',codeCont):
                        variable_list.append(f"InstruPrint[\"{name}\"]=*{value};")
                    else:
                        variable_list.append(f"InstruPrint[\"{name}\"]={value};")
                
                variables_string = " ".join(variable_list)
                print_statement = f'''{variables_string} std::cout << "Block{i}:" << InstruPrint.dump() << std::endl; InstruPrint.clear();\n'''
                
                if self.starts_with_return(bodyRemovedCode):
                    code_list = code.split("\n")
                    return_index= [i for i in range(len(code_list)) if "return" in code_list[i]][0]
                    if return_index>0 and re.search('[a-zA-Z]', code_list[0]):
                        while return_index-1 >=0 and not re.search('[a-zA-Z]', code_list[return_index-1]):
                            return_index = return_index - 1
                        
                    code_list[return_index] = " " * self.GetIndent(code) + print_statement + "\n" + code_list[return_index]
                    InstruCode = "\n".join(code_list)
                elif code.count("{") == code.count("}"): 
                    if (self.RemainingCode.split(f"BLOCK{i}-END")[-1].strip().startswith('else') or 
                        self.RemainingCode.split(f"BLOCK{i+1}-START")[-1].strip().startswith('else')) and code.strip().endswith("}"):
                        InstruCode = code[:code.rfind('}')] + "\n" + print_statement + code[code.rfind('}'):] 
                    else: InstruCode = code + "\n" + print_statement
                    
                else:
                    code_list = code.split("\n")
                    letter_index = [i for i in range(len(code_list)) if bool(re.search('[a-zA-Z]', code_list[i]))][-1]
                    if i + 1 >= len(self.blockedCode_list):
                        if i >= len(self.blockedCode_list): IndentCode = self.GetIndent(self.blockedCode_list[-1])
                        else: IndentCode = self.GetIndent(self.blockedCode_list[i])
                    else: IndentCode = self.GetIndent(self.blockedCode_list[i+1])
                    code_list[letter_index] = code_list[letter_index]  +  "\n" + " " * IndentCode + print_statement
                    InstruCode = "\n".join(code_list)
                    
            elif len(variables)>0 and not any(item in ["Integer", "String", "gson","InstruPrint", "else"] for item in variables):
                print_statement = f'''std::cout << "Block{i}:" << InstruPrint.dump() << std::endl; InstruPrint.clear();'''
                InstruCode = " "*self.GetIndent(code)+print_statement + "\n" + code
            else: 
                InstruCode = code
                
            if ("if(" in code or "while(" in code) and (code.count("{")!= code.count("}") and "{" in code) and "else if" not in code:
                print_statement = "\n" + f'''std::cout << "Block{i}: CONDITION" << std::endl;'''
                InstruCode = print_statement + "\n" + InstruCode
            self.allVariable.extend(valuePrint)
            return InstruCode
        
    def generate_json_conversion_code(self, var_name, code_content):
        patterns = {
            r'(std::)?vector\s*<\s*std::bitset<\d+>\s*>': lambda name: f'''InstruPrint = {{"{name}", [&{name}]() {{ std::vector<std::string> vec; for (const auto& bs : {name}) vec.push_back(bs.to_string()); return vec; }}()}};'''
        }
        for pattern, generator in patterns.items():
            if re.search(rf'{pattern}\s+{re.escape(var_name)}\b', code_content):
                return generator(var_name)
        return ""

    def extract_variables(self, node):
        variables = []
        if node.type == 'identifier' and node.parent and node.parent.type not in ['function_definition',
                                                                                  'class_definition',
                                                                                  'method_declaration']:
            # This is a variable, not a function or class name
            variables.append(node.text.decode('utf8'))

        for child in node.children:
            variables.extend(self.extract_variables(child))
        return variables

    def splitMethod(self,s):
        split_s = s.split("(",1)[0]
        return split_s.strip()
    
    
    def extract_parentheses_content(self, s):
        pattern = re.compile(r'\((.*?)\)')
        matches = pattern.findall(s)
        
        def find_nested(s):
            stack = []
            result = []
            i = 0
            while i < len(s):
                if s[i] == '(':
                    stack.append(i)
                elif s[i] == ')' and stack:
                    start = stack.pop()
                    if not stack:  # only add if it's the outermost
                        result.append(s[start + 1:i])
                i += 1
            return result

        all_matches = []
        for match in matches:
            all_matches.append(match)
            nested_matches = find_nested(match)
            all_matches.extend(nested_matches)
        
        return all_matches
    
    def removeBody(self, code_str) -> str:
        keyMatch = self.toolUse.contains_keywords(code_str)
        code_str_list = [code for code in code_str.split("\n") if code.strip()]
        if keyMatch == False or (keyMatch == True and len(code_str_list)==1): return code_str
        indent_list = [self.GetIndent(code) for code in code_str_list] 
        key_index = [i for i in range(len(code_str_list)) if self.toolUse.contains_keywords(code_str_list[i])][0] 


        last_key_indent = -1
        for j in range(key_index, len(indent_list)):
            if key_indent > indent_list[j]:
                last_key_indent = j
                break
        
        return "\n".join(code_str_list[:key_index]) + "\n" + "\n".join(code_str_list[last_key_indent+1:])
    
    def IfWhile(self, s, tag=False):
        keywords = ["if", "while", "elif"]
        pattern = r'\b(' + '|'.join(keywords) + r')\b|\b(' + '|'.join(keywords) + r')(?=\W)'

        match = re.search(pattern, s)
        if tag:
            matchKeys = re.finditer(pattern, s)
            KeyIndex = [(match.group(1), match.start(1), match.end(1)) for match in matchKeys]
            return KeyIndex
        return bool(match)
    
    def ForStmt(self, s, tag=False):
        keywords = ["for"]
        pattern = r'\b(' + '|'.join(keywords) + r')\b|\b(' + '|'.join(keywords) + r')(?=\W)'

        match = re.search(pattern, s)
        if tag:
            matchKeys = re.finditer(pattern, s)
            KeyIndex = [(match.group(1), match.start(1), match.end(1)) for match in matchKeys]
            return KeyIndex
        return bool(match)

    def preprocess_string(self, s):
        if '''f"''' in s and s.count("(") == s.count(")"): return s 
        def repl(match):
            return '_' * (match.end() - match.start())
        return re.sub(r'(["\']).*?\1', repl, s)

    

    def IsVariable_initialized(self, code_str, var_name, allCode):
        data_type = ['byte', 'short', 'int', 'long', 'float', 'double', 'char', 'boolean', 'String']
        code_list = code_str.split("\n")
        code_str_t = ""
        for code in code_list:
            code_str_t = code_str_t + code.split("(",1)[0]

        pattern = r'\b(' + '|'.join(data_type) + r')\b|\b(' + '|'.join(data_type) + r')(?=\W)'
        match = re.search(pattern, code_str_t)
        pattern3 = rf"\b{var_name}\s=\snew "
        if bool(match):  
            pattern1 = rf"\b{var_name}\s*,"
            pattern2 = rf"\b{var_name}\s*;"
            
            if (re.search(pattern1, code_str_t) or re.search(pattern2, code_str_t)) and ("(" not in code_str_t and ")" not in code_str_t):
                return False  
            else: return True  
        elif re.search(pattern3, code_str_t) and var_name not in allCode :
            return False 
        return True

    
    def conditionExtraction_python(self, KeyIndex, Code):
        Code = Code.rstrip()
        Condition_list = []
        for index in KeyIndex:
            End = index[2]
            for i in range(End, len(Code)):
                char = Code[i]
                if char == ":" and i+1 == len(Code):
                    Condition_list.append(Code[End: i])
                    break
        return Condition_list

    def conditionExtraction_java_cpp(self, KeyIndex, Code):
        T_code = self.preprocess_string(Code)

        Condition_list = []
        stack = []
        for index in KeyIndex:
            End = index[2]
            if T_code[End] != "(": continue
            for i in range(End, len(T_code)):
                char = T_code[i]
                if char == "(":
                    stack.append(i)
                elif char == ")" and stack:
                    start = stack.pop()
                    if not stack:
                        Condition_list.append(Code[start: i+1])
                        break

        return Condition_list
    
    
    def return_python_functionName_codeFuse(self, code_content):
        tree = ast.parse(code_content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                return function_name
    
    def remove_comments_java_cpp(self, java_code):

        pattern = r'(".*?"|\'.*?\'|//.*?$|/\*.*?\*/)'

        def replace_func(match):
            if match.group(0).startswith(("//", "/*")):
                return ""  # Remove comments
            else:
                return match.group(0)  # Keep string literals

        cleaned_code = re.sub(pattern, replace_func, java_code, flags=re.DOTALL | re.MULTILINE)
        return cleaned_code

    def pythonCodeType(self, pythonCode):
        pattern = re.compile(r'^\s*(def|class|if|elif|else|for|while|try|except|finally|with)\b')
        return bool(pattern.match(pythonCode.strip()))
    