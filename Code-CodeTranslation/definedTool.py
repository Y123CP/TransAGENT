import re

class definedTool:
    def indexFinder(self, keywords, sourceStr):
        pattern = r'\b(' + '|'.join(keywords) + r')\b|\b(' + '|'.join(keywords) + r')(?=\W)'
        matchKeys = re.finditer(pattern, sourceStr)
        KeyIndex = [(match.group(1), match.start(1), match.end(1)) for match in matchKeys]
        return KeyIndex
    
    def preprocess_string(self, s):
        def repl(match):
            return '_' * (match.end() - match.start())
        return re.sub(r'(["\']).*?\1', repl, s)
    
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
    
    

    def contains_keywords(self,s:str, tag="keyMatch"):

        if tag == "keyMatch":
            keywords = ["if", "else", "while", "for", "try", "catch", "exception", "do", "switch", "case", "return", "break", "except",
                        "elif", "continue"]
            pattern = r'\b(' + '|'.join(keywords) + r')\b|\b(' + '|'.join(keywords) + r')(?=\W)'

            # 搜索字符串
            match = re.search(pattern, s)

            # 检查是否找到匹配
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

    
    def functionExtraction(self, target_lang, trans_Code, method_signature, Metric=False, remove_comments=True):
        
        if target_lang == "java":
            if remove_comments: trans_Code = self.remove_comments_java_cpp(trans_Code)
            codeBlock = []
            left_brack_list = []
            right_brack_list = []
            Start_Tag = False
            JavaCode_list = trans_Code.split("\n")
            for index in range(len(JavaCode_list)):
                line = JavaCode_list[index]
                if Metric: CONDITION = method_signature in line and "class" not in line  and ("{" in line or (len(JavaCode_list)> index+1 and "{" in JavaCode_list[index+1])) and "static" in line
                else: CONDITION  =  self.find_and_replace(method_signature.strip(), line)[0] and "class" not in line  and ("{" in line or (len(JavaCode_list)> index+1 and "{" in JavaCode_list[index+1])) and ("static" in line or "public" in line)
                if CONDITION:  
                    line = self.find_and_replace(method_signature.strip(), line)[1]
                    Start_Tag = True
                    if "static" not in line and "public" in line:  
                        line_str = line.replace("public", "public static")
                        codeBlock.append(line_str)
                    else:
                        code_split = [code for code in line.split("(")[0].split(" ") if code.strip()]
                        if len(code_split) == 2:  # 只有一个 identify + method Name
                            new_line = "static " + line.strip() + "\n"
                            codeBlock.append(new_line)
                        else:codeBlock.append(line)
                    left_brack_count = line.count("{")
                    left_brack_list.extend(["{"] * left_brack_count)
                    right_brack_count = line.count("}")
                    right_brack_list.extend(["}"] * right_brack_count)
                    if len(left_brack_list) == len(right_brack_list) and len(left_brack_list)!=0: break
                    continue
                if Start_Tag:
                    codeBlock.append(line)
                    
                    left_brack_count = line.count("{")
                    left_brack_list.extend(["{"] * left_brack_count)
                    right_brack_count = line.count("}")
                    right_brack_list.extend(["}"] * right_brack_count)
                    if len(left_brack_list) == len(right_brack_list):
                        break
            codeBlock_str = "\n".join(codeBlock)
            return codeBlock_str
        elif target_lang == "python":
            if remove_comments: trans_Code = self.remove_comments_python(trans_Code)
            PythonCode_list = [code for code in trans_Code.split("\n") if code.strip()]
            codeBlock = []
            Start_Tag = False
            for code in PythonCode_list:
                if self.find_and_replace(method_signature.strip(), code)[0] and "def " in code:
                    code = self.find_and_replace(method_signature.strip(), code)[1]
                    def_indent = self.GetIndent(code) # obtain the indent of the line
                    codeBlock.append(code)
                    Start_Tag = True
                    continue
                if Start_Tag:
                    code_indent = self.GetIndent(code)
                    if def_indent < code_indent or "# ----" in code:
                        codeBlock.append(code)
                    else:break
            codeBlock_str = "\n".join(codeBlock)
            return codeBlock_str
        elif target_lang == "cpp":
            if remove_comments: trans_Code = self.remove_comments_java_cpp(trans_Code)
            codeBlock = []
            left_brack_list = []
            right_brack_list = []
            Start_Tag = False
            CppCode_list = trans_Code.split("\n")
            for index in range(len(CppCode_list)):
                line = CppCode_list[index]
                if Metric: CONDITION = method_signature in line and  ("{" in line or (len(CppCode_list)> index+1 and "{" in CppCode_list[index+1]))
                else: CONDITION = self.find_and_replace(method_signature.strip(), line)[0] and ("{" in line or (len(CppCode_list)> index+1 and "{" in CppCode_list[index+1]))
                if CONDITION:
                    line = self.find_and_replace(method_signature.strip(), line)[1]
                    Start_Tag = True
                    codeBlock.append(line)
                    left_brack_count = line.count("{")
                    left_brack_list.extend(["{"] * left_brack_count)
                    right_brack_count = line.count("}")
                    right_brack_list.extend(["}"] * right_brack_count)
                    if len(left_brack_list) == len(right_brack_list) and len(left_brack_list)!=0: break
                    continue
                if Start_Tag:
                    codeBlock.append(line)
                    left_brack_count = line.count("{")
                    left_brack_list.extend(["{"] * left_brack_count)
                    right_brack_count = line.count("}")
                    right_brack_list.extend(["}"] * right_brack_count)
                    if len(left_brack_list) == len(right_brack_list):
                        break
            codeBlock_str = "\n".join(codeBlock)
            return codeBlock_str
        

    def find_and_replace(self, str1, str2):
        def normalize(s):
            return re.sub(r'_', '', s).lower()
        normalized_to_original = {}
        words = re.findall(r'\b\w+\b', str2)
        for word in words:
            normalized_word = normalize(word)
            if normalized_word not in normalized_to_original:
                normalized_to_original[normalized_word] = word

        normalized_str1 = normalize(str1)

        for normalized, original in normalized_to_original.items():
            if normalized == normalized_str1:
                pattern = re.compile(r'\b{}\b'.format(re.escape(original)), re.IGNORECASE)
                replaced_str2 = pattern.sub(str1, str2, count=1)
                return True, replaced_str2
        
        return False, str2

    def GetIndent(self, s):
        return len(s) - len(s.lstrip(' '))  
    

    def remove_comments_java_cpp(self, java_code):
        pattern = r'(?s)"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//.*?$|/\*.*?\*/'

        def replace_func(match):
            if match.group(0).startswith(("//", "/*")):
                return ""  # Remove comments
            else:
                return match.group(0)  # Keep string literals
        cleaned_code = re.sub(pattern, replace_func, java_code, flags=re.MULTILINE)
        return cleaned_code
    
    def remove_comments_python(self,code):
        code = re.sub(r"(?<!\\)#.*", "", code)
        def replacer(match):
            if match.group(1).startswith(("'''", '"""')) and match.group(1).endswith(("'''", '"""')):
                return ""
            else: 
                return match.group(0)
        code = re.sub(r'(\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\")', replacer, code, flags=re.DOTALL)
        
        return code

if __name__ == "__main__":
    target_lang = "java"
    trans_code = '''
    import java.util.ArrayList;

static ArrayList<Integer> move_first ( ArrayList<Integer> test_list ) {
    int last = test_list.remove(test_list.size() - 1);
    test_list.add(0, last);
    return test_list;
}'''
    method_signature = 'moveFirst'
    tool = definedTool()
    tool.functionExtraction(target_lang, trans_code, method_signature)