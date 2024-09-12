import glob
import os
import ast
import json
import re
import shutil
import subprocess
from definedTool import definedTool
from tree_sitter import Language, Parser

def extract_method_names_java_python(code, language):
    PARSER_LOCATION = f"./CodeTranslation/JavaEnv/{language}_parser.so"
    LANGUAGE = Language(PARSER_LOCATION, language)
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node
    def extract_methods(node):
        methods = []
        for child in node.children:
            if child.type == 'function_definition' or child.type == 'method_declaration':
                for c in child.children:
                    if c.type == 'identifier':
                        methods.append(c.text.decode('utf8'))
            methods.extend(extract_methods(child))
        return methods
    return extract_methods(root_node)

# def extract_method_names_cpp(code, language="cpp"):
#     PARSER_LOCATION = f"./CodeTranslation/JavaEnv/{language}_parser.so"
#     LANGUAGE = Language(PARSER_LOCATION, language)
#     parser = Parser()
#     parser.set_language(LANGUAGE)
#     tree = parser.parse(bytes(code, "utf8"))
#     root_node = tree.root_node

#     def extract_methods(node):
#         methods = []
#         if node.type == 'function_definition':
#             func_declarator = next((child for child in node.children if child.type == 'function_declarator'), None)
#             if func_declarator:
#                 identifier = next((child for child in func_declarator.children if child.type == 'identifier'), None)
#                 if identifier:
#                     methods.append(identifier.text.decode('utf8'))
#         for child in node.children:
#             methods.extend(extract_methods(child))
#         return methods

#     return extract_methods(root_node)

def extract_method_names_cpp(code, language="cpp"):
    PARSER_LOCATION = f"./CodeTranslation/JavaEnv/{language}_parser.so"
    LANGUAGE = Language(PARSER_LOCATION, language)
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node


    def extract_methods(node):
        methods = []

        if node.type == 'function_definition':
            func_declarator = next((child for child in node.children if child.type == 'function_declarator'), None)
            if func_declarator:
    
                identifier = next((child for child in func_declarator.children if child.type in ['identifier', 'field_identifier']), None)
                if identifier:
                    methods.append(identifier.text.decode('utf8'))
        for child in node.children:
            methods.extend(extract_methods(child))
        return methods

    return extract_methods(root_node)


class Manually:
    def __init__(self,source_lan, target_lan):
        # cpp, java, python
        self.source_lan = source_lan
        self.target_lan = target_lan
        self.toline = TOLINE()
        self.toolUse = definedTool()
        
    def file_analyzer(self):
        output_list = []
        # sourceCodeFiles = "./CodeTranslation/ExperimentDataset/Manually/new_one_function.json"
        sourceCodeFiles = f"./CodeTranslation/ExperimentDataset/Manually/{self.source_lan}2{self.target_lan}.jsonl"
        with open(sourceCodeFiles, 'r', encoding='utf-8') as f:
            jsonData = json.load(f)
        
        count = 0
        for Data in jsonData:
            output_dict = {"source_Lan":str, "source_code_str":None,"source_code_block":"", "target_Lan":str, "reference_code":str}
            
            source_code = Data[f"{self.source_lan}_code"]
            target_code = Data[f"{self.target_lan}_code"]
            # if "static int bobHappiness(int n, int k, int q, int[][] stu" not in target_code: continue

            
            if self.source_lan == "python":
                source_code_str, source_Method_Signature, source_allMS, source_import = self.extract_function_python(source_code)
            elif self.source_lan == "java" or self.source_lan == "cpp":
                source_code_str, source_Method_Signature, source_allMS, source_import= self.extract_function_java_cpp(source_code, self.source_lan)
                        

            if self.target_lan == "python":
                reference_code, target_Method_Signature, target_allMS,target_import = self.extract_function_python(target_code)
            else:
                reference_code, target_Method_Signature, target_allMS,target_import= self.extract_function_java_cpp(target_code, self.target_lan)
            
            
            if len(reference_code) == 0 or len(source_code_str) == 0: 
                count = count + 1
                continue
            
            output_dict['source_Lan'] = self.source_lan + "###" + self.getFileName(source_Method_Signature, self.source_lan)
            output_dict['source_code_str'] = source_code_str
            output_dict['target_Lan'] = self.target_lan + "###" + self.getFileName(target_Method_Signature, self.target_lan)
            output_dict['reference_code'] = reference_code
            
            # output_dict['ori_target_code_shell'] = target_code_shell  # code_shell is the target_code
            # output_dict['ori_source_code_shell'] = source_code_shell
            
            output_dict['target_method_signature'] = target_Method_Signature # Method_signature is the target_code
            output_dict['source_method_signature'] = source_Method_Signature
            
            output_dict['target_allMS'] = target_allMS 
            output_dict['source_allMS'] = source_allMS
            
            output_dict['source_import'] = source_import
            output_dict['target_import'] = target_import
            
            # output_dict['input_output_value'] = input_output_value
            # output_dict['all_input_output_value'] = input_output_value
            if self.target_lan in ["java","cpp"]: output_dict['commentTag'] = "//"
            elif self.target_lan == "python": output_dict['commentTag'] = "#"
            
            output_list.append(output_dict)
        print(count)
        return output_list
        

    def getFileName(self, Method_Signature, lang):
        if lang == "java":
            fileName = Method_Signature + "_Test.java"
        elif lang == "python":
            fileName = Method_Signature + ".py"
        elif lang == "cpp":
            fileName = Method_Signature + ".cpp"
        return fileName
    
    def extract_function_python(self, codeCont):
        functionBlock = []
        # code_shell = ""

        source = "\n".join([code for code in codeCont.split("\n") if code.strip()])
        lines = source.splitlines()
        
        import_info = []
        for line in lines:
            if "import " in line:
                import_info.append(line)
                
        extractMS = extract_method_names_java_python(source, "python")
        if len(extractMS) >1: 
            # print("python_MS\n",codeCont)
            return "", "", "", ""
        Method_Signature = extractMS[0]
        

        # tree = ast.parse(source)
        # for node in ast.walk(tree):
        #     if isinstance(node, ast.FunctionDef) and node.name == Method_Signature: # "".join(node.name.split("_")).lower() == file_name:
        #         start_line = node.lineno - 1 
        #         end_line = node.end_lineno  
        #         functionBlock_list = lines[start_line:end_line]
        functionBlock_list = lines
        firstLine_indent = self.toline.GetIndent(functionBlock_list[0])
        functionBlock_list_new = [self.remove_leading_spaces(code, firstLine_indent) for code in functionBlock_list]

   
        for line in functionBlock_list_new:
            if Method_Signature in line:
                allMS = line.replace("self,","")
                break
        functionBlock  = "\n".join(functionBlock_list_new).replace("self,","")
        return functionBlock, Method_Signature, allMS, "\n".join(import_info)


    def remove_leading_spaces(self, Str, num_spaces):
        leading_spaces = ' ' * num_spaces
        if Str.startswith(leading_spaces):
            return Str[num_spaces:]
        return Str

    def remove_Javacomments(self, java_code):
        pattern = r'(".*?"|\'.*?\'|//.*?$|/\*.*?\*/)'

        def replace_func(match):
            if match.group(0).startswith(("//", "/*")):
                return ""  # Remove comments
            else:
                return match.group(0)  # Keep string literals
        cleaned_code = re.sub(pattern, replace_func, java_code, flags=re.DOTALL | re.MULTILINE)
        return cleaned_code


    def extract_function_java_cpp(self,codeCont,lang):
        
        cont = "\n".join([code for code in codeCont.split('\n') if code.strip()])
        
        import_info = []
        if lang == "java":
            extractMS_t = extract_method_names_java_python(cont, "java")
            
            cont_list = cont.split('\n')
            extractMS = []
            for ms in extractMS_t:
                for cont_line in cont_list:
                    if "static " in cont_line and ms in cont_line:
                        extractMS.append(ms)
            if len(extractMS_t) == 1 and len(extractMS) == 0:
                cont = cont.replace("public ", "public static ")
                extractMS = extractMS_t
            import_info.append("import java.util.*;")
            import_info.append("import java.lang.*;")
            for line in cont.split("\n"):
                if "import " in line and ";" in line:
                    import_info.append(line)
        elif lang == "cpp":
            extractMS = extract_method_names_cpp(cont, "cpp")
            import_info.append("#include <iostream>")
            for line in cont.split("\n"):
                if "#include" in line or ("using " in line and ";" in line):
                    import_info.append(line)
            if "#include <vector>" not in cont: import_info.append("#include <vector>")
            if "#include <unordered_map>" not in cont: import_info.append("#include <unordered_map>")
            if "using namespace std;" not in cont: import_info.append("using namespace std;")
                    
        javaCode = self.remove_Javacomments(cont)

        if len(extractMS) >1 or len(extractMS) == 0: 
            return "", "", "", ""
        Method_Signature = extractMS[0]

        functionBlock = self.toolUse.functionExtraction(lang, javaCode, Method_Signature,remove_comments=True)
        for line in functionBlock.split("\n"):
            if Method_Signature in line:
                target_allMS = line
                break
            
        return functionBlock, Method_Signature, target_allMS, "\n".join(import_info)
    

class TOLINE:
    
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
 
    def Reformat_java_lineCode(self, codeBlock:str) -> str:
        codeBlock = codeBlock.replace(" (", "(")
        codeBlock = self.Reformat_for_if(codeBlock)
        KeyIndex = self.contains_keywords(codeBlock, "toLine")
        if len(KeyIndex)>0: codeBlock = self.ReFormatBracket(KeyIndex, codeBlock, True)

        functionBlock_str = self.InnerFunction_java(codeBlock)
        return functionBlock_str


    def InnerFunction_java(self, codeCont:str) -> str:
        functionBlock = []
        codeCont_list = codeCont.split("\n")
        j = -1
        for i in range(len(codeCont_list)):
            if j >= i: continue
            line_code = codeCont_list[i]
            sub_block = []
            if line_code.strip().endswith("(") or line_code.endswith("="):
                sub_block.append(line_code)
                codeIndent = self.GetIndent(line_code)
                j = i
                while True:
                    if j + 1 == len(codeCont_list):
                        break
                    if self.GetIndent(codeCont_list[j + 1]) - codeIndent > 2:
                        sub_block.append(codeCont_list[j + 1])
                    else:
                        functionBlock.append(" "*codeIndent + " ".join("\n".join(sub_block).split()))
                        sub_block.clear()
                        break
                    j = j + 1
            else: functionBlock.append(line_code)
        return "\n".join(functionBlock)
    
    def ReFormatBracket(self, KeyIndex, Code, Tag=False):
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

        if Tag:
            LineOfCondition = [" ".join(code.split()) for code in Condition_list]
            for i in range(len(Condition_list)):
                Code = Code.replace(Condition_list[i], LineOfCondition[i])
            return Code
        elif len(Condition_list) == 0 and Tag:
            return Code

        else: return Condition_list

    # 为 if/ else if Stmt --> if/else if {Stmt}
    def Reformat_for_if(self, codeBlock_str:str, time=1) -> str:
        codeBlock = codeBlock_str.split("\n")
        functionBlock_list = []
        j = -1
        for i in range(len(codeBlock)):
            if i <= j: continue
            lineCode = codeBlock[i]
            if (lineCode.strip().startswith("for") or lineCode.strip().startswith(
                    "else if") or lineCode.strip().startswith("if") or lineCode.strip().startswith("while") or lineCode.strip().startswith("} else if")) and "{" not in lineCode:
                if lineCode.strip().endswith(")") and "{" not in codeBlock[i+1]: 
                    functionBlock_list.append(lineCode + " {")
                    count = 1
                    codeIndent = self.GetIndent(lineCode)
                    j = i
                    while True:
                        j = j + 1
                        if j == len(codeBlock):
                            break
                        if self.GetIndent(codeBlock[j]) <= codeIndent and not re.fullmatch(r"^}+$", codeBlock[j].strip()):
                            functionBlock_list.append(" " * codeIndent + "}"*count)
                            j = j -  1
                            break
                        elif codeBlock[j].strip().endswith(")"): 
                            functionBlock_list.append(codeBlock[j] + " {")
                            count = count + 1
                        else:
                            functionBlock_list.append(codeBlock[j])
                elif lineCode.strip().endswith(";"):  
                    KeyIndex = self.contains_keywords(lineCode, "addBracket")
                    condition_list = self.ReFormatBracket(KeyIndex, lineCode)
                    new_lineCode = lineCode
                    for condition in condition_list:
                        new_lineCode = new_lineCode.replace(condition, condition + " {\n" + " " * self.GetIndent(lineCode))
                    functionBlock_list.append(new_lineCode + "\n" + "}" * len(condition_list))
                else:
                    functionBlock_list.append(lineCode)

            else:
                functionBlock_list.append(lineCode)

        return "\n".join(functionBlock_list)
    
    def preprocess_string(self, s):
        def repl(match):
            return '_' * (match.end() - match.start())

        return re.sub(r'(["\']).*?\1', repl, s)
    
    def contains_keywords(self,s:str, tag="keyMatch"):

        if tag == "keyMatch":
            keywords = ["if", "else", "while", "for", "try", "catch", "exception", "do", "switch", "case", "return", "break", "except",
                        "elif", "continue"]
            pattern = r'\b(' + '|'.join(keywords) + r')\b|\b(' + '|'.join(keywords) + r')(?=\W)'

            match = re.search(pattern, s)
            return bool(match)

        else:
            if tag == "toLine":
                keywords = ["if", "else if", "while", "for", "elif", "return"]
            elif tag == "addBracket":
                keywords = ["if", "else if", "while", "for", "elif"]
            pattern = r'\b(' + '|'.join(keywords) + r')\b|\b(' + '|'.join(keywords) + r')(?=\W)'

            match = re.search(pattern, s)
            matchKeys = re.finditer(pattern, s)
            KeyIndex = [(match.group(1), match.start(1), match.end(1)) for match in matchKeys]
            return KeyIndex    
    
    def Reformat_python_lineCode(self, codeCont:str) -> str:
        functionBlock = []
        codeCont_list = codeCont.split("\n")
        j = -1
        for i in range(len(codeCont_list)):
            if j >= i: continue
            lineCode = codeCont_list[i]
            left_bracket_list = []
            right_bracket_list = []
            if lineCode.strip().endswith("("):
                left_bracket_list.extend(["("] * lineCode.count('('))
                right_bracket_list.extend([")"] * lineCode.count(')'))
                for j in range(i+1, len(codeCont_list)):
                    code = codeCont_list[j]
                    left_bracket_list.extend(["("] * code.count('('))
                    right_bracket_list.extend([")"] * code.count(')'))
                    if len(left_bracket_list) == len(right_bracket_list):
                        sub_code = "".join(codeCont_list[i:j+1])
                        codeIndent = self.GetIndent(sub_code)
                        functionBlock.append(" "*codeIndent + " ".join(sub_code.split()))
                        break
            else:
                functionBlock.append(lineCode)
        return "\n".join(functionBlock)

    def GetIndent(self, s):
        return len(s) - len(s.lstrip(' '))

    def FormatCode(self, filepath,lang):
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
            
if __name__ == "__main__":
    tasks = ["python2java"]
    for task in tasks:
        source = task.split('2')[0]
        trans = task.split('2')[-1]
        Mainually_in = Manually(source,trans)
        Mainually_in.file_analyzer()