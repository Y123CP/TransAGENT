
TestFG_FurtherRefine= '''
Source {source_lang} Code is as follows:
{source_code}

Translated {target_lang} Code is as follows:
{trans_code}

@Command: Please follow the two steps below to correct the {target_lang} Code and make it equivalent to the {source_lang} Code!
Step 1: Generate descriptive comments for the {source_lang} Code.
Step 2: Based on the comments, Fill in the Correct Code between `--1--` and `--2--` in {target_lang} Code!
'''

TestFG_Refine= '''
Source {source_lang} Code is as follows:
{source_code}

Translated {target_lang} Code is as follows:
{trans_code}

Given the Input at `--1--`:
{Input_dict}
Actual Output at `--2--` in the {target_lang} code:
{Actually_Output}
But Expected Output:
{Expected_Output}

@Command: Please follow the two steps below to fix the {target_lang} Code and make it equivalent to the {source_lang} Code!
Step 1: Check for the issues in the {target_lang} code based on the Actual Output at position `--2--`.
Step 2: Fix the {target_lang} code and make it equivalent to the {source_lang} Code!!!!
'''



MessageConver_parameterType = '''
### Example
{Example}

### Task
Error Message and Location:
{ErrorMessage}
Code that needs to be fixed:
`{ErrorLine}`
@Command: Please transform the error message into a user-friendly action suggestion (only one sentence) for fixing the Code.
Pay Attention: What needs to be fixed is the type in:  {ErrorLine}.
@Output:
'''

MessageConver_parameterType_java = '''
Error Message and Location:
Throw `incompatible types: char[] cannot be converted to String`, at `System.out.println(f_gold("a*c*a".toCharArray(), 5));`
Code that needs to be fixed:
`static String f_gold(String string, int l) `
@Command: Please transform the error message into a user-friendly action suggestion (only one sentence) for fixing the Code.
Pay Attention: What needs to be fixed is the type in:  static String f_gold(String string, int l).
@Output:
```To fix this error, you should change the `String` in the `static String f_gold(String string, int l)` to the `chat[]` ```
'''

MessageConver_parameterType_cpp = '''
Error Message and Location:
Throw `cannot convert ‘std::vector<int>’ to ‘int*’`, at `int x = maxScoreSubseq(0, arr1);`
Code that needs to be fixed:
`int maxScoreSubseq(int n, int arr[]) {`
@Command: Please transform the error message into a user-friendly action suggestion (only one sentence) for fixing the Code.
Pay Attention: What needs to be fixed is the type in:  int maxScoreSubseq(int n, int arr[]) {.
@Output:
```To fix this error, you should change the `int` in the `int maxScoreSubseq(int n, int arr[])` to the `vector<int>` ```
'''

MessageConver_not_parameterType = '''
### Example
{Example}

### Task
{lang} Code is as follows:
{function_code}

Error Message is as follows:
{ErrorMessage}
Error Location is as follows:
{ErrorLine}
@Command: Analyze the above error Message based on the Error Location (i.e., <Buggy Line>), and then transform it into a user-friendly action suggestion (only one sentence).
@Output:
'''

MessageConver_not_parameterType_java = '''
Java Code is as follows:
public static boolean initializeGraph() {
        ArrayList<ArrayList<Integer>> adjList = new ArrayList<>(); // <Buggy Line>
        adjList.add(new ArrayList<>()); 
        return adjList.get(0).add(2); 
    }
    
Error Message is as follows:
Throw `cannot find symbol `class ArrayList``, at `ArrayList<ArrayList<Integer>> adjList = new ArrayList<>();`
Error Location is as follows:
ArrayList<ArrayList<Integer>> adjList = new ArrayList<>();
@Command: Analyze the above error Message based on the Error Location (i.e., <Buggy Line>), and then transform it into a user-friendly action suggestion (only one sentence).
@Output:
```'ArrayList' is not imported. To fix this error, you should import the `import java.util.ArrayList;` at the beginning of your code`. ```
'''
MessageConver_not_parameterType_cpp = '''
Cpp Code is as follows:
int main() {
    std::cout << value; // <Buggy Line>
    return 0;
}

Error Message is as follows:
‘value’ was not declared in this scope
Error Location is as follows:
std::cout << value;
@Command: Analyze the above error Message based on the Error Location (i.e., <Buggy Line>), and then transform it into a user-friendly action suggestion (only one sentence).
@Output:
``` To fix this error, you should declare the 'value' variable before using it. ```
'''
MessageConver_not_parameterType_python = '''
Python Code is as follows:
def f_gold ( a ) :
    area = ( math.pi * a * a ) / 4 # <Buggy Line>
    return area
    
Error Message is as follows: 
name 'math' is not defined
Error Location is as follows:
area = ( math.pi * a * a ) / 4
@Command: Analyze the above error Message based on the Error Location (i.e., <Buggy Line>), and then transform it into a user-friendly action suggestion (only one sentence).
@Output:
```'math' module has not been imported. To fix this error, you should import `math`.
'''


TestInputGen = '''
### Example
{Example}

### Task
## {source_lang}_code
{source_code}

## Analyze the given {source_lang}_code, and then generate test input with high-level line coverage for it. Be aware that the generated test input may cause type overflow issues. The generated Test Inputs are in the format of [Input_1] [Input_2]... [Input_5] and should be wrapped in ``` ```. ##
## Test Inputs
'''

TestInputGen_validTest = '''
### Example
{Example}

### Task
## {source_lang}_code
{source_code}

## Analyze the given {source_lang}_code, and then generate test input with high-level line coverage for it. The generated Test Inputs are in the format of [Input_1] [Input_2]... [Input_5] and should be wrapped in ``` ```. ##
## Test Inputs
'''

MainFuncGen = '''
### Example
{Example}

### Task
## {source_lang}_code
{source_code}

## Test Inputs
{Test_Inputs}

## Analyze the `{source_lang}_code`, and then generate a correct Main_Function to execute the given Test Inputs via the `{methodSig}` (Includes the import statements needed to execute this code). The generated Main_Function should be wrapped within ``` ```!!!! ##
## Output
'''

inputOutputArchive = '''
### Example
{Example}

### Task
## {lang}_code
{code}
    
## After running the printed result is as follows
{run_result}

## Please match the input in {lang}_code with the printed result. The match result should be wrapped within ``` ```.##
## Output
'''


Code2CodeTrans_trans_inputOutput = '''
### Example
{source_Lan} Code is as follows:
{source_example}

Given the Test Cases:
{Input_output_example}

## Translate the `{source_Lan} Code` into the equivalent {target_Lan} function code (only one equivalent function), and ensure the translated function code can pass all given test cases. NOTE: The translated {target_Lan} function Code should use  ``{Method_signature_example}`` as method name (Include the necessary import statement), and be wrapped within ``` ```!!! ##
## Output:
```{target_Lan}
{target_example}
```

### Task
{source_Lan} Code is as follows:
{To_be_translated}

Given the Test Cases:
{Input_output}

## Translate the `{source_Lan} Code` into the equivalent {target_Lan} function code (only one equivalent function), and ensure the translated function code can pass all given test cases. NOTE: The translated {target_Lan} function Code should use  ``{Method_signature}`` as method name (Include the necessary import statement), and be wrapped within ``` ```!!! ##
## Rule: When translating the `{source_Lan} Code` into the `{target_Lan} Code`, please follow the Control Flow Graph (CFG) structure of the source code. 
## Output:
'''

Code2CodeTrans_trans_inputOutput_Compilerepair_our = '''
### Example
{Example}

### Task
{target_lang} Code:
{target_code}

Given test cases:
{TestCase}

Fix Suggestion:
{ErrorMessage}
@Command: Repair the buggy line (marked {commentTag} <Buggy Line>) in the buggy {target_lang} code according to the fix suggestion. The generated {target_lang} Code should use  ``{Method_signature}`` as the method name, and be wrapped within ``` ```.
@Output:
'''


Code2CodeTrans_trans_inputOutput_Compilerepair = '''
{Example}

## {target_lang}_code
{target_code}

## Given test case
{TestCase}

## Error Message: 
{ErrorMessage}
## Repair the buggy line (marked {commentTag} <Buggy Line>) in the buggy {target_lang} code according to the given Error Message. The generated {target_lang} Code should use  ``{Method_signature}`` as the method name, and be wrapped within``` ```. ###
## Output
'''


Code2CodeTrans_inputOutput_RunTimerepair = '''
{Example}

## {target_lang}_code
{target_code}

## Given test case
{TestCase}

## Execution Result
### Error Type: Runtime Error
### Error Line: {Line}
### Error Message: 
{ErrorMessage}
### Repair the buggy {target_lang} code according to the given `## Execution Result`. The generated {target_lang} Code should use  ``{Method_signature}`` as the method name, and be wrapped within``` ```. ###
## Output
'''



Code2CodeTrans_inputOutput_AllInRunrepair = '''
{Example}

## {target_lang}_code
{target_code}

## Given test case
{TestCase}

### Error Message: {ErrorMessage}, expected output and actual output are note equal!
## Repair the buggy {target_lang} code according to the given Error Message. The generated {target_lang} Code should use  ``{Method_signature}`` as the method name, and be wrapped within``` ```. ###
## Output
'''


CodeMapping = '''
{Example}

## {target_lang}_code
{target_code}

## {source_lang}_code
{To_be_translated}
    
## Analyze the relationship between {target_lang}_code and {source_lang}_code, and then carefully map the {target_lang} BLOCK code (marked {commentTag} BLOCK) to the {source_lang} code. Note that the mapped code must can be find in {target_lang}_code or {source_lang}_code. ##
## Output
'''

#################### Trans Coder dataset START #########################


Code2CodeTrans_trans = '''
### Example
{source_Lan} Code is as follows:
{source_example}

## Translate the `{source_Lan} Code` into the equivalent {target_Lan} function code (only one equivalent function). NOTE: The translated {target_Lan} function Code should use  ``{Method_signature_example}`` as method name (Include the necessary import statement), and be wrapped within ``` ```!!! ##
## Output:
```{target_Lan}
{target_example}
```

### Task
{source_Lan} Code is as follows:
{To_be_translated}

## Translate the `{source_Lan} Code` into the equivalent {target_Lan} function code (only one equivalent function). NOTE: The translated {target_Lan} function Code should use  ``{Method_signature}`` as method name (Include the necessary import statement), and be wrapped within ``` ```!!! ##
## Output:
'''




FunctionDescription = '''
{Example}

## {source_lang}_code
{Code}

## Please generate a function description for the code between ---1--- and ---2--- in {source_lang} code. The generated description can only be for the code snippet between --1-- and --2--, and the description should be clear enough to be implemented in {target_lang} code. ##
## Output
'''


RGCode = '''
{Example}

## {Lang}_code
{Code}

## Generate code between between --1-- and --2-- according to the Function Description and Input-Output Example. Return the compilable `## {Lang}_code` (Include the necessary import statement), and be wrapped within``` ```.##
## Output
'''



##### Trans_Code_st_Code_shell: Used for Execute Java Code ##############
Java_Code_Shell = '''
import java.util. *;
import java.util.stream.*;
import java.lang.*;
import javafx.util.Pair;
public class Demo{
    $CODEFILL$
}
'''
##########################################################################


################################################ Example ############################################
########### Code Trans one-shot #############
python_trans_code_example = '''
def f_gold ( x ) :
    return ( - ( ~ x ) ) ;
'''
java_trans_code_example = '''
static int f_gold ( int x ) {
  return ( - ( ~ x ) ) ;
}
'''
cpp_trans_code_example = '''
int f_gold ( int x ) {
  return ( - ( ~ x ) );
}
'''

example_VariableType = '''
## Variable_Type:
`x` type is int
`f_gold` return type is int
'''


########### Code CompilRepair one-shot #############
python_Compilerepair = '''
## Python_code
def f_gold ( a ) :
    area = ( math.pi * a * a ) / 4 # <Buggy Line>
    return area

## Given test case
Input: (a = 77)
Expected_output: (area = 4656.625)

## Error Message: Error: Unresolved reference 'math'
## Repair the buggy line (marked # <Buggy Line>) in the buggy Python code according to the given Error Message. The generated Python Code should use  ``def f_gold`` as the method name, and be wrapped within``` ```. ###
## Output 
```python
import math
def f_gold ( a ) :
    area = ( math.pi * a * a ) / 4
    return area
```
'''

java_Compilerepair = '''
## Java_code
static float f_gold ( int a ) {
    float area = ( float ) (PI + a + a ) / 4 ; // <Buggy Line
    return area ;
}

## Given test case
Input: (a = 77)
Expected_output: (area = 4656.625)

## Error Message: Cannot resolve symbol 'PI'
## Repair the buggy line (marked // <Buggy Line>) in the buggy Java code according to the given Error Message. The generated Java Code should use  ``f_gold`` as the method name, and be wrapped within``` ```. ###
## Output 
```java
static float f_gold ( int a ) {
    float area = ( float ) (Math.PI + a + a ) / 4 ;
    return area ;
}
```
'''

cpp_Compilerepair = '''
## Cpp_code
double f_gold(int n) {
    return (3.0 * n) / (4.0 * (n * m) - 1);
}

## Given test case
Input: (n = 10)
Expected_output: (-30)

## Error Message: ‘m’ was not declared in this scope
## Repair the buggy line (marked // <Buggy Line>) in the buggy Cpp_code according to the given Error Message. The generated Cpp_code should use  ``f_gold`` as the method name, and be wrapped within``` ```. ###
## Output 
```cpp
double f_gold(int n) {
    int m =0;
    return (3.0 * n) / (4.0 * (n * m) - 1);
}
```
'''


python_Compilerepair_our = '''
Python Code:
def f_gold ( a ) :
    area = ( math.pi * a * a ) / 4 # <Buggy Line>
    return area

Given test cases:
Input: (a = 77)
Expected_output: (area = 4656.625)

Fix Suggestion: 
```To fix this error, you should import `math`.
@Command: Repair the buggy line (marked # <Buggy Line>) in the buggy Python code according to the fix suggestion. The generated Python Code should use  ``f_gold`` as the method name, and be wrapped within``` ```.
@Output:
```python
import math
def f_gold ( a ) :
    area = ( math.pi * a * a ) / 4
    return area
```
'''

java_Compilerepair_our = '''
Java Code:
static ArrayList<Integer> f_gold(int a) {  // <Buggy Line>
    ArrayList<Integer> numbers = new ArrayList<>(); 
    for (int i = 0; i < a; i++) {
        numbers.add(i);
    }
    return numbers.get(1);
}

Given test cases:
Input: a = 77
Expected_output: 1

Fix Suggestion:
```To fix this error, you should import `java.util.ArrayList````
@Command: Repair the buggy line (marked // <Buggy Line>) in the buggy Java code according to the fix suggestion. The generated Java Code should use  ``f_gold`` as the method name, and be wrapped within``` ```.
@Output: 
```java
import java.util.ArrayList;

static ArrayList<Integer> f_gold(int a) {  
    ArrayList<Integer> numbers = new ArrayList<>(); 
    for (int i = 0; i < a; i++) {
        numbers.add(i);
    }
    return numbers.get(1);
}
```
'''

cpp_Compilerepair_our = '''
Cpp Code:
double f_gold(int n) {
    return (3.0 * n) / (4.0 * (n * m) - 1); // <Buggy Line>
}

Given test cases:
Input: (n = 10)
Expected_output: (-30)

Fix Suggestion: 
```To fix this error, you should decalre the ‘m’ berfor use it.```
@Command: Repair the buggy line (marked // <Buggy Line>) in the buggy Cpp code according to the fix suggestion. The generated Cpp Code should use  ``f_gold`` as the method name, and be wrapped within``` ```.
@Output:
```cpp
double f_gold(int n) {
    int m =0;
    return (3.0 * n) / (4.0 * (n * m) - 1);
}
```
'''
########### Code RunRepair one-shot #############
python_AllInRunrepair = '''
## Python_code
def f_gold ( a ) :
    area = ( math.pi + a + a ) / 4
    return area
    
## Given test case
Input: (a = 77)
Expected_output: (area = 4656.625)

### Error Message: Expected Output: area = 4656.625, Acutal Output: 39.285, expected output and actual output are note equal!
## Repair the buggy Python code according to the given Error Message. The generated Python Code should use  ``def f_gold`` as the method name, and be wrapped within``` ```. ###
## Output
```python
def f_gold ( a ) :
    area = ( math.pi * a * a ) / 4
    return area
```
'''

java_AllInRunrepair = '''
## Java_code
static float f_gold ( int a ) {
    float area = ( float ) ( Math . PI + a + a ) / 4 ;
    return area ;
}

## Given test case
Input: (a = 77)
Expected_output: (area = 4656.625)

### Error Message: Expected Output: area = 4656.625, Acutal Output: 39.285, expected output and actual output are note equal!
## Repair the buggy Java code according to the given Error Message. The generated Java Code should use  ``f_gold`` as the method name, and be wrapped within``` ```. ###
## Output
```java
static float f_gold ( int a ) {
    float area = ( float ) ( Math . PI * a * a ) / 4 ;
    return area ;
}
```
'''


cpp_AllInRunrepair = '''
## Cpp_code
float f_gold ( int a ) {
    float area = (float)(M_PI + a + a ) / 4 ;
    return area ;
}

## Given test case
Input: (a = 77)
Expected_output: (area = 4656.625)

### Error Message: Expected Output: area = 4656.625, Acutal Output: 39.285, expected output and actual output are note equal!
## Repair the buggy Cpp_code according to the given Error Message. The generated Cpp_code should use  ``f_gold`` as the method name, and be wrapped within``` ```. ###
## Output
```cpp
float f_gold ( int a ) {
    float area = (float)(M_PI * a * a ) / 4 ;
    return area ;
}
```
'''


python_RunTimerepair =  '''
## Python_code
def f_gold ( a ) :
    area = a + " " + a
    return area

## Given test case
Input: 77
Expected_output: "77 77"

## Execution Result
### Error Type: Runtime Error
### Error Line: area = a + " " + a
### Error Message: TypeError: unsupported operand type(s) for +: 'int' and 'str'
### Repair the buggy Python code according to the given `## Execution Result`. The generated Python Code should use  ``def f_gold`` as the method name, and be wrapped within``` ```. ###
## Output
```python
def f_gold ( a ) :
    area = str(a) + " " + str(a)
    return area
```
'''


java_RunTimerepair =  '''
## Java_code
static int sum1(int[] numbers) {
    int total = 0;
    for (int i = 0; i <= numbers.length; i++) { 
        total += numbers[i];
    }
    return total;
}

## Given test case
Input: {1, 2, 3, 4, 5}
Expected_output: 15

## Execution Result
### Error Type: Runtime Error
### Error Line: total += numbers[i];
### Error Message: Exception in thread "main" java.lang.ArrayIndexOutOfBoundsException
### Repair the buggy Java code according to the given `## Execution Result`. The generated Python Code should use  ``sum1`` as the method name, and be wrapped within``` ```. ###
## Output
```java
static int sum1(int[] numbers) {
    int total = 0;
    for (int number : numbers) {
        total += number;
    }
    return total;
}
```
'''

cpp_RunTimerepair =  '''
## Cpp_code
bool f_gold(char str[], int k) {
  int n = strlen(str);
  int c = 0;
  for (int i = 0; i < k; i++)
    if (str[n - i - 1] == '0')
      c++;
  int* p = nullptr;
  *p = 42;
  return (c == k);
}

## Given test case
Input: str = "111010100", k=2
Expected_output: 1

## Execution Result
### Error Type: Runtime Error
### Error Line: *p = 42;
### Error Message: Segmentation fault.
### Repair the buggy Cpp_code according to the given `## Execution Result`. The generated Cpp_code should use  ``f_gold`` as the method name, and be wrapped within``` ```. ###
## Output
```cpp
bool f_gold(char str[], int k) {
  int n = strlen(str);
  int c = 0;
  for (int i = 0; i < k; i++)
    if (str[n - i - 1] == '0')
      c++;
  int* p = nullptr;
  return (c == k);
}
```
'''


########### Code input_output one-shot #############



java_repaired = '''
static float f_gold ( int a ) {
    float area = ( float ) ( Math . PI * a * a ) / 4 ;
    return area ;
}
'''

python_repaired = '''
static float f_gold ( int a ) {
    float area = ( float ) ( Math . PI + a + a ) / 4 ;
    return area ;
}
'''

##########################################


########### Code mapping one-shot #############
python2java_mapping = '''
## Python_code:
def f_gold ( a ) : # BLOCK0
# ----
    area = ( math.pi * a * a ) / 4 # BLOCK1
    return area # BLOCK1
# ----
    
## Java_code:
static float f_gold ( int a ) {
    return ( float ) (PI + a + a ) / 4 ;
}

## Analyze the relationship between Python_code and Java_code, and then carefully map the Python BLOCK code (marked # BLOCK) to the Java code. Note that the mapped code must can be find in Python_code or Java_code. ##
## Output
BLOCK0: 
```python
def f_gold ( a ) :
```
Corresponding Java Code:
```java
static float f_gold ( int a ) {
```
BLOCK1: 
```python
area = ( math.pi * a * a ) / 4
return area
```
Corresponding Java Code:
```java
return ( float ) (PI + a + a ) / 4 ;
```
'''

java2python_mapping = '''
## Java_code
static float f_gold ( int a ) { // BLOCK0
// ----
    area = ( float ) (PI + a + a ) / 4; // BLOCK1
    return area; // BLOCK1
// ----
}

## Python_code
def f_gold ( a ) :
    return ( math.pi * a * a ) / 4  
    
## Analyze the relationship between Java_code and Python_code, and then carefully map the Java BLOCK code (marked // BLOCK) to the Python code. Note that the mapped code must can be find in Java_code or Python_code. ##
## Output
BLOCK0: 
```java
static float f_gold ( int a ) {
```
Corresponding Python Code:
```python
def f_gold ( a ) :
```
BLOCK1: 
```java
area = ( float ) (PI + a + a ) / 4;
return area;
```
Corresponding Python Code:
```python
return ( math.pi * a * a ) / 4  
```
'''


cpp2python_mapping = '''
## cpp_code
float f_gold ( int a ) { // BLOCK0
// ----
    area = ( float ) (PI + a + a ) / 4; // BLOCK1
    return area; // BLOCK1
// ----
}

## Python_code
def f_gold ( a ) :
    return ( math.pi * a * a ) / 4  
    
## Analyze the relationship between cpp_code and Python_code, and then carefully map the cpp BLOCK code (marked // BLOCK) to the Python code. Note that the mapped code must can be find in cpp_code or Python_code. ##
## Output
BLOCK0: 
```cpp
float f_gold ( int a ) {
```
Corresponding Python Code:
```python
def f_gold ( a ) :
```
BLOCK1: 
```cpp
area = ( float ) (PI + a + a ) / 4;
return area;
```
Corresponding Python Code:
```python
return ( math.pi * a * a ) / 4  
```
'''

cpp2java_mapping = '''
## cpp_code
float f_gold ( int a ) { // BLOCK0
// ----
    area = ( float ) (PI + a + a ) / 4; // BLOCK1
    return area; // BLOCK1
// ----
}
    
## Java_code:
static float f_gold ( int a ) {
    return ( float ) (PI + a + a ) / 4 ;
}

## Analyze the relationship between cpp_code and Java_code, and then carefully map the cpp BLOCK code (marked # BLOCK) to the Java code. Note that the mapped code must can be find in cpp_code or Java_code. ##
## Output
BLOCK0: 
```cpp
float f_gold ( int a ) {
```
Corresponding Java Code:
```java
static float f_gold ( int a ) {
```
BLOCK1: 
```cpp
area = ( float ) (PI + a + a ) / 4;
return area;
```
Corresponding Java Code:
```java
return ( float ) (PI + a + a ) / 4 ;
```
'''

java2cpp_mapping = '''
## Java_code
static float f_gold ( int a ) { // BLOCK0
// ----
    area = ( float ) (PI + a + a ) / 4; // BLOCK1
    return area; // BLOCK1
// ----
}

## cpp_code
float f_gold ( int a ) { 
    area = ( float ) (PI + a + a ) / 4;
    return area;
}

## Analyze the relationship between Java_code and cpp_code, and then carefully map the Java BLOCK code (marked # BLOCK) to the cpp code. Note that the mapped code must can be find in Java_code or cpp_code. ##
## Output
BLOCK0: 
```java
static float f_gold ( int a ) { 
```
Corresponding cpp Code:
```cpp
float f_gold ( int a ) { 
```

BLOCK1: 
```java
area = ( float ) (PI + a + a ) / 4;
return area; 
```
Corresponding cpp Code:
```cpp
area = ( float ) (PI + a + a ) / 4;
return area;
```
'''
python2cpp_mapping = '''
## Python_code:
def f_gold ( a ) : # BLOCK0
# ----
    area = ( math.pi * a * a ) / 4 # BLOCK1
    return area # BLOCK1
# ----

## cpp_code
float f_gold ( int a ) { 
    area = ( float ) (PI + a + a ) / 4;
    return area;
}

## Analyze the relationship between Python_code and cpp_code, and then carefully map the Python BLOCK code (marked # BLOCK) to the cpp code. Note that the mapped code must can be find in Python_code or cpp_code. ##
## Output
BLOCK0: 
```python
def f_gold ( a ) :
```
Corresponding cpp Code:
```cpp
float f_gold ( int a ) { 
```

BLOCK1: 
```python
area = ( math.pi * a * a ) / 4
return area
```
Corresponding cpp Code:
```cpp
area = ( float ) (PI + a + a ) / 4;
return area;
```
'''




###################### Block Translation Example #############
python_block_code_example = '''
def f_gold ( x ) : 
    m = 1 ;
    while ( x & m ) : 
        x = x ^ m 
        m <<= 1 
    x = x ^ m 
    return x
'''



java_block_code_example = '''
public static int f_gold(int x) { 
    int m = 1; 
    while ((x & m) != 0) { 
        x = x ^ m; 
        m <<= 1; 
    }
    x = x ^ m; 
    return x;
}
'''

cpp_block_code_example = '''
int f_gold(int x) {
    int m = 1;
    while (x & m) {
      x = x ^ m;
      m <<= 1;
    }
    x = x ^ m;
    return x;
}
'''

python_block_runRepair_correct = '''
def f_gold ( a ) :
    area = ( math.pi * a * a ) / 4
    return area
'''




##########################################
Input_output_example = '''
Input: (x)= (96,)
Expected_output: 97
'''

###################################
TestInputGen_python = '''
## Python_code
def f_gold ( a ) :
    area = ( math.pi * a * a ) / 4
    return area

## Analyze the given Python_code_code, and then generate test input with high-level line coverage for it. Be aware that the generated test input may cause type overflow issues. The generated Test Inputs are in the format of [Input_1] [Input_2]... [Input_5] and should be wrapped in ``` ```. ##
## Test Inputs
```
input1: a=0
input2: a=1e6
input3: a=-1
input4: a=10
input5: a=5
'''
TestInputGen_java = '''
## Java_code
static float f_gold ( int a ) {
    float area = (float) ((Math.PI * a * a ) / 4);
    return area;
}

## Analyze the given Java_code, and then generate test input with high-level line coverage for it. Be aware that the generated test input may cause type overflow issues. The generated Test Inputs are in the format of [Input_1] [Input_2]... [Input_5] and should be wrapped in ``` ```. ##
## Test Inputs
```
input1: a=0
input2: a=1e6
input3: a=-1
input4: a=10
input5: a=5
```
'''
TestInputGen_cpp = '''
## Cpp_code
float f_gold(int a) {
  float area = (M_PI * a * a) / 4.0;
  return area;
}

## Analyze the given Cpp_code, and then generate test input with high-level line coverage for it. Be aware that the generated test input may cause type overflow issues. The generated Test Inputs are in the format of [Input_1] [Input_2]... [Input_5] and should be wrapped in ``` ```. ##
## Test Inputs
```
input1: a=0
input2: a=1e6
input3: a=-1
input4: a=10
input5: a=5
```
'''

MainFuncGen_python = '''
## Python_code
def f_gold ( a ) :
    ...
    
## Test Inputs
input1: a=0
input2: a=16

## Analyze the `Python_code`, and then generate a correct Main_Function to execute the given Test Inputs via the `f_gold`. The generated Main_Function should be wrapped within ``` ```. ##
## Output
```
if __name__ == "__main__":
    print(f_gold(0))
    print(f_gold(16))
```
'''

MainFuncGen_java = '''
## Java_code
static float f_gold ( int a ) {
...
    
## Test Inputs
input1: a=0
input2: a=16
## Analyze the `Java_code`, and then generate a correct Main_Function to execute the given Test Inputs via the `f_gold` in . The generated Main_Function should be wrapped within ``` ```. ##
## Output
```
public static void main(String[] args) {
    System.out.println(f_gold(0));
    System.out.println(f_gold(6));
    }
'''

MainFuncGen_cpp = '''
## Cpp_code
float f_gold(int a) {
...
    
## Test Inputs
input1: a=0
input2: a=16

## Analyze the `Cpp_code`, and then generate a correct Main_Function to execute the given Test Inputs via the `f_gold`. The generated Main_Function should be wrapped within ``` ```. ##
## Output
```
int main() {
    cout << f_gold(0) << endl;
    cout << f_gold(16) << endl;
}
```
'''

inputOutputArchive_java = '''
## Java_code
static float f_gold ( int a ) {
...
public static void main(String[] args) {
    System.out.println(f_gold(0));
    System.out.println(f_gold(6));
    System.out.println(f_gold(-1));
    System.out.println(f_gold(10));
    System.out.println(f_gold(5));
}
    
## After running the printed result is as follows
0.0
28.274334
0.7853982

## Please match the input in Java_code with the printed result. The match result should be wrapped within ``` ```.##
## Output
```
input1: a=0
output1: 0.0
----
input2: a=6
output2: 28.274334
----
input3: a=-1
output3: 0.7853982
'''


inputOutputArchive_cpp = '''
## Cpp_code
float f_gold(int a) {
...
int main() {
    cout << f_gold(0) << endl;
    cout << f_gold(16) << endl;
    cout << f_gold(-1) << endl;
    cout << f_gold(10) << endl;
    cout << f_gold(5) << endl;
}
    
## After running the printed result is as follows
0.0
28.274334
0.7853982

## Please match the input in Cpp_code with the printed result. The match result should be wrapped within ``` ```.##
## Output
```
input1: a=0
output1: 0.0
----
input2: a=6
output2: 28.274334
----
input3: a=-1
output3: 0.7853982
'''

inputOutputArchive_python = '''
## Python_code
def f_gold ( a ) :
...
if __name__ == "__main__":
    print(f_gold(0))
    print(f_gold(16))
    print(f_gold(-1))
    print(f_gold(10))
    
## After running the printed result is as follows
0.0
28.274334
0.7853982

## Please match the input in Python_code with the printed result. The match result should be wrapped within ``` ```.##
## Output
```
input1: a=0
output1: 0.0
----
input2: a=6
output2: 28.274334
----
input3: a=-1
output3: 0.7853982
'''



inputOutputType_cpp = '''
## Cpp_code
float f_gold(int a) {
 // ...
    
## Test Cases
input1: a=0
output1: 0
---
input2: a=6
output2: 28.27
---
input3: a=-1
output3: 0.78

## Please infer the Input-Output Type in the Test Cases according to the Cpp_code. The generated Main_Function should be wrapped within ``` ```. ##
## Output
```
input1: int a=0
output1: (float) 0
---
input2: int a=6
output2: (float) 28.27
---
input3: int a=-1
output3: (float) 0.78
```
'''



inputOutputType_java = '''
## Cpp_code
static float f_gold ( int a ) {
 // ...
    
## Test Cases
input1: a=0
output1: 0
---
input2: a=6
output2: 28.27
---
input3: a=-1
output3: 0.78

## Please infer the Input-Output Type in the Test Cases according to the Java_code. The generated Main_Function should be wrapped within ``` ```. ##
## Output
```
input1: int a=0
output1: (float) 0
---
input2: int a=6
output2: (float) 28.27
---
input3: int a=-1
output3: (float) 0.78
```
'''

transMap_python2java = '''
### Python Code
def f_gold ( x ) : # --- py stmt 1
    m = 1 ; # --- py stmt 2
    while ( x & m ) : # --- py stmt 3
        x = x ^ m # --- py stmt 4
        m <<= 1 # --- py stmt 5
    x = x ^ m # --- py stmt 6
    return x # --- py stmt 7

### Java Code
public static int f_gold(int x) { 
    int m = 1; 
    while ((x & m) != 0) {
        x = x ^ m;
        m <<= 1;
    } 
    x = x ^ m; 
    return x;
}
### Match the Python Code to the Java Code statement by statement.
### Response
```java
public static int f_filled(int x) { // --- py stmt 1
    int m = 1; // --- py stmt 2
    while ((x & m) != 0) { // --- py stmt 3
        x = x ^ m; // --- py stmt 4
        m <<= 1; // --- py stmt 5
    }
    x = x ^ m; // --- py stmt 6
    return x; // --- py stmt 7
}
```
'''

transMap_python2cpp = '''
### Python Code
def f_gold ( x ) : # --- py stmt 1
    m = 1 ; # --- py stmt 2
    while ( x & m ) : # --- py stmt 3
        x = x ^ m # --- py stmt 4
        m <<= 1 # --- py stmt 5
    x = x ^ m # --- py stmt 6
    return x # --- py stmt 7

### C++ Code
int f_gold(int x) {
    int m = 1;
    while (x & m) {
      x = x ^ m;
      m <<= 1;
    }
    x = x ^ m;
    return x;
}

### Match the Python Code to the C++ Code statement by statement.
### Response
```cpp
int f_filled(int x) { // --- py stmt 1
    int m = 1; // --- py stmt 2
    while (x & m) { // --- py stmt 3
        x = x ^ m; // --- py stmt 4
        m <<= 1; // --- py stmt 5
    }
    x = x ^ m; // --- py stmt 6
    return x; // --- py stmt 7
}
```
'''


transMap_cpp2python = '''
### C++ Code
int f_filled(int x) { // --- cpp stmt 1
    int m = 1; // --- cpp stmt 2
    while (x & m) { // --- cpp stmt 3
        x = x ^ m; // --- cpp stmt 4
        m <<= 1; // --- cpp stmt 5
    }
    x = x ^ m; // --- cpp stmt 6
    return x; // --- cpp stmt 7
}

### Python Code
def f_gold ( x ) :
    m = 1 ;
    while ( x & m ) :
        x = x ^ m
        m <<= 1
    x = x ^ m
    return x
    
### Match the C++ Code to the Python Code statement by statement.
### Response
```python
def f_gold ( x ) : # --- cpp stmt 1
    m = 1 ; # --- cpp stmt 2
    while ( x & m ) : # --- cpp stmt 3
        x = x ^ m # --- cpp stmt 4
        m <<= 1 # --- cpp stmt 5
    x = x ^ m # --- cpp stmt 6
    return x # --- cpp stmt 7
```
'''


transMap_cpp2java = '''
### C++ Code
int f_filled(int x) { // --- cpp stmt 1
    int m = 1; // --- cpp stmt 2
    while (x & m) { // --- cpp stmt 3
        x = x ^ m; // --- cpp stmt 4
        m <<= 1; // --- cpp stmt 5
    }
    x = x ^ m; // --- cpp stmt 6
    return x; // --- cpp stmt 7
}

### Java Code
public static int f_gold(int x) { 
    int m = 1; 
    while ((x & m) != 0) {
        x = x ^ m;
        m <<= 1;
    } 
    x = x ^ m; 
    return x;
}

### Match the C++ Code to the Java Code statement by statement.
### Response
```java
public static int f_filled(int x) { // --- cpp stmt 1
    int m = 1; // --- cpp stmt 2
    while ((x & m) != 0) { // --- cpp stmt 3
        x = x ^ m; // --- cpp stmt 4
        m <<= 1; // --- cpp stmt 5
    }
    x = x ^ m; // --- cpp stmt 6
    return x; // --- cpp stmt 7
}
```
'''


transMap_java2python = '''
### Java Code
public static int f_filled(int x) { // --- java stmt 1
    int m = 1; // --- java stmt 2
    while ((x & m) != 0) { // --- java stmt 3
        x = x ^ m; // --- java stmt 4
        m <<= 1; // --- java stmt 5
    }
    x = x ^ m; // --- java stmt 6
    return x; // --- java stmt 7
}

### Python Code
def f_gold ( x ) :
    m = 1 ;
    while ( x & m ) :
        x = x ^ m
        m <<= 1
    x = x ^ m
    return x
    
### Match the Java Code to the Python Code statement by statement.
### Response
```python
def f_gold ( x ) : # --- java stmt 1
    m = 1 ; # --- java stmt 2
    while ( x & m ) : # --- java stmt 3
        x = x ^ m # --- java stmt 4
        m <<= 1 # --- java stmt 5
    x = x ^ m # --- java stmt 6
    return x # --- java stmt 7
```
'''

transMap_java2cpp = '''
### Java Code
public static int f_gold(int x) { // --- java stmt 1
    int m = 1; // --- java stmt 2
    while ((x & m) != 0) { // --- java stmt 3
        x = x ^ m; // --- java stmt 4
        m <<= 1; // --- java stmt 5
    }
    x = x ^ m; // --- java stmt 6
    return x; // --- java stmt 7
}

### C++ Code
int f_gold(int x) {
    int m = 1;
    while (x & m) {
      x = x ^ m;
      m <<= 1;
    }
    x = x ^ m;
    return x;
}

### Match the Java Code to the C++ Code statement by statement.
### Response
```cpp
int f_gold(int x) { // --- java stmt 1
    int m = 1; // --- java stmt 2
    while (x & m) { // --- java stmt 3
        x = x ^ m; // --- java stmt 4
        m <<= 1; // --- java stmt 5
    }
    x = x ^ m; // --- java stmt 6
    return x; // --- java stmt 7
}
```
'''
transMap = '''
{Example}

## {source_lang}_code
{To_be_translated}

## {target_lang}_code
{target_code}
    
### Match the {source_lang} Code to the {target_lang} Code statement by statement.
### Response
'''


