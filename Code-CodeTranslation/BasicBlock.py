import subprocess
import os
import sys
import pygraphviz as pgv
import glob
import re


def mainStart(codePath, methodSig):
    # dotPath generation
    graph = dotGeneration(codePath,methodSig)
    
    # dotPath = "./gen_cfg/LEXICOGRAPHICAL_MAXIMUM_SUBSTRING_STRINGCFG/2-cfg.dot"
    # graph = pgv.AGraph(dotPath)
    BasicBlockNode = iterate_graph(graph)
    
    blockCode = BasicBlockSplit(codePath, graph, BasicBlockNode)
    return blockCode
    

def dotGeneration(codePath,methodSig):
    current_path = os.getcwd()
    fileName = os.path.basename(codePath).split(".")[0]
    BinPath = os.path.join(os.path.dirname(codePath), fileName+".bin")
    CFGPath = os.path.join(os.path.dirname(codePath), fileName+"CFG")


    commandjava1 = f"joern-parse {codePath} --output {BinPath}"
    commandjava2 = f"joern-export {BinPath} --repr cfg --out {CFGPath}"
    subprocess.run(commandjava1,shell=True, check=True)
    subprocess.run(commandjava2,shell=True, check=True)
    
    dotPaths = [file for file in glob.glob(CFGPath+"/*") if ".dot" in file]
    for dotPath in dotPaths:
        graph = pgv.AGraph(dotPath)
        if graph.name.replace("_","").lower() == methodSig.replace("_","").lower():
            return graph
     
    
def IterateStart(graph, BasicBlockNode, start_node, tag=False):
    block1, block2, nodes = ConBasic(graph, start_node, tag)
    BasicBlockNode.append(block1)
    if block2 in BasicBlockNode and len(block2) != 0: return BasicBlockNode
    BasicBlockNode.append(block2)
    
    for start_node in nodes:
        BasicBlockNode = IterateStart(graph, BasicBlockNode, start_node, tag=True)
    return BasicBlockNode


def iterate_graph(graph):
    DealedNode = []
    
    BasicBlockNode = []
    stack = [node for node in graph.nodes() if graph.in_degree(node) == 0 and graph.out_degree(node) == 1]
    BasicBlockNode.append(stack.copy())

    while stack:
        start_node = stack.pop()
        DealedNode.append(start_node)
        block, successors, next_successors = ConBasic(graph, start_node, [start_node] not in BasicBlockNode)
        if len(block): BasicBlockNode.append(block)
        if len(successors): BasicBlockNode.append(successors)
        for node in next_successors:
            if node not in DealedNode and node not in stack: 
                stack.append(node) 

    return BasicBlockNode





def ConBasic(graph, current_node, tag):
    block = []
    if tag: block.append(current_node)
    while True:
        successors = list(graph.successors(current_node)) 
        if len(successors) == 1:
            next_node = successors[0]
            if graph.in_degree(next_node) == 1 and graph.out_degree(next_node) == 1:
                block.append(next_node)
                current_node = next_node
            else:  
                next_successors = list(graph.successors(next_node)) 
                return block, successors, next_successors
        else: 
            return block, [], successors
            

def BasicBlockSplit(codePath, graph, BasicBlockNode):
    PlaceHolder = "$$$"
    fileEnds = os.path.basename(codePath).split(".")[-1]
    commentTag = " # " if fileEnds == "py" else " // "
    
    LineNumberBlock = []
    for Block in BasicBlockNode:
        if len(Block) == 0: continue
        block = []
        for node in Block:
            node_obj = graph.get_node(node)
            labelAttr = node_obj.attr['label']
            numberMatch = re.search(r'<SUB>(\d+)</SUB>', labelAttr)
            if not numberMatch: continue
            lineNumebr = int(numberMatch.group(1))
            block.append(lineNumebr)
        if block not in LineNumberBlock: LineNumberBlock.append(block)
    
    with open(codePath, 'r', encoding='utf-8') as f:
        codeCont = f.read()
    codeCont = " \n" + codeCont
    codeContList = codeCont.split("\n")
    
    DealedLine = []

    singleBlock = [block[0] for block in LineNumberBlock if len(block) == 1]
    for line in singleBlock:
        DealedLine.append(line)
        codeContList[line] = codeContList[line]  +commentTag + "BLOCK$SINGLE$" 

    for lineBlock in LineNumberBlock:
        if len(lineBlock) == 1: continue
        lineBlock_add = []
        for line in list(set(lineBlock)):
            if line not in DealedLine: lineBlock_add.append(line)
        if len(lineBlock_add) == 0: continue
        lineBlock_add = sorted(lineBlock_add)   
        start_line = lineBlock_add[0]
        end_line = lineBlock_add[-1]
        for line in range(start_line, end_line+1):
            codeContList[line] = codeContList[line]  + commentTag + "BLOCK$SAME$"+str(start_line)
        DealedLine.extend(lineBlock_add)
        
    startNumber = 0
    index = -1
    for i in range(len(codeContList)):
        if i < index: continue
        if "BLOCK$SINGLE$" in codeContList[i]: 
            codeContList[i] = codeContList[i].replace("$SINGLE$", str(startNumber)) + f"\n{commentTag} ----"
            startNumber = startNumber + 1
        elif "BLOCK$SAME$" in codeContList[i]: 
            tag1 = codeContList[i].split("BLOCK$SAME$")[-1]
            codeContList[i] = codeContList[i].replace("$SAME$"+tag1, str(startNumber))
            index = i 
            while True:
                index = index + 1
                tag2 = codeContList[index].split("BLOCK$SAME$")[-1]
                if "BLOCK$SAME$" in codeContList[index] and tag1 == tag2:
                    codeContList[index] = codeContList[index].replace("$SAME$"+tag1, str(startNumber))
                else: break
            codeContList[index-1] = codeContList[index-1]+ f"\n{commentTag} ----"
            startNumber = startNumber + 1
    return "\n".join(codeContList)


    
if __name__ == "__main__":
    codePath = "./gen_cfg/LEXICOGRAPHICAL_MAXIMUM_SUBSTRING_STRING.py"
    blockCode = mainStart(codePath, "f_filled")
    print(blockCode)
 