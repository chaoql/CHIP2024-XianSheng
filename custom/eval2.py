import json
from llama_index.core import PromptTemplate
from tqdm import tqdm
import tools
import solution
from pipeline import rag
import pipeline.readData as readData
from llama_index.core.response_synthesizers.type import ResponseMode
import os
from dotenv import load_dotenv, find_dotenv
from dotenv import dotenv_values
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core import Settings
from custom.glmfz import ChatGLM


def load_all_evalData(submit_path, tasknum):
    with open(submit_path, 'r', encoding='utf-8', errors='ignore') as file:
        data = file.readlines()
    result = []
    # 遍历数据中的每个案例
    for i, case in enumerate(data):
        dataValue = case.split("@")
        result.append(dataValue[tasknum].rstrip("\n"))
    return result


def evalTask1(result1, train_path):
    with open(train_path, 'r', encoding='utf-8', errors='ignore') as file:
        train_data = json.load(file)
    all_score = 0.0
    for i, res in enumerate(result1):
        oknums = 0
        score = 0.0
        resList = res.split(";")
        trainList = train_data[i]['信息抽取能力-核心临床信息'].split(";")
        allnums = len(trainList)
        for re in resList:
            if re in trainList:
                oknums += 1
        score = oknums / allnums
        all_score += score
        print(f"案例{i + 1}: {score}")
    print(f"task1总得分: {all_score}/50")
    return all_score


def evalTask2(result, train_path):
    with open(train_path, 'r', encoding='utf-8', errors='ignore') as file:
        train_data = json.load(file)
    all_score = 0.0
    for i, res in enumerate(result):
        oknums = 0
        score = 0.0
        resList = res.split(";")
        trainList = train_data[i]['病机答案'].split(";")
        for re in resList:
            if re in trainList:
                oknums += 1
        score = oknums / (len(trainList) + len(resList) - oknums)
        all_score += score
        print(f"案例{i + 1}: {score}")
    print(f"task2总得分: {all_score}/50")
    return all_score


def evalTask3(result, train_path):
    with open(train_path, 'r', encoding='utf-8', errors='ignore') as file:
        train_data = json.load(file)
    all_score = 0.0
    for i, res in enumerate(result):
        oknums = 0
        score = 0.0
        resList = res.split(";")
        trainList = train_data[i]['证候答案'].split(";")
        allnums = len(trainList)
        for re in resList:
            if re in trainList:
                oknums += 1
        score = oknums / (allnums + len(resList) - oknums)
        all_score += score
        print(f"案例{i + 1}: {score}")
    print(f"task3总得分: {all_score}/50")
    return all_score


# def evalTask4(result, train_path):
#     with open(train_path, 'r', encoding='utf-8', errors='ignore') as file:
#         train_data = json.load(file)
#     all_score = 0.0
#     for i, res in enumerate(result):
#         oknums = 0
#         score = 0.0
#         resList = res.split(";")
#         trainList = train_data[i]['证候答案'].split(";")
#         allnums = len(trainList)
#         for re in resList:
#             if re in trainList:
#                 oknums += 1
#         score = oknums / (allnums + len(resList) - oknums)
#         all_score += score
#         print(f"案例{i + 1}: {score}")
#     print(f"task3总得分: {all_score}/50")
#     return all_score


_ = load_dotenv(find_dotenv())  # 导入环境
config = dotenv_values(".env")
# 加载嵌入模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-zh-v1.5",
    cache_folder="./BAAI/",
    embed_batch_size=128,
    local_files_only=True,  # 仅加载本地模型，不尝试下载
    device="cuda",
)
# 加载大模型
Settings.llm = ChatGLM(
    api_key=config["GLM_KEY"],
    model="GLM-4",
    api_base="https://open.bigmodel.cn/api/paas/v4/",
    is_chat_model=True,
)

# 设置参数
with_hyde = False  # 是否采用假设文档
persist_dir = "storeQ"  # 向量存储地址
hybrid_search = False  # 是否采用混合检索
with_LLMrerank = True
top_k = 10
rerank_top_k = 3
response_mode = ResponseMode.COMPACT  # 最佳实践为为TREE_SUMMARIZE
hybrid_mode = "OR"
submitPath = "submit/eval1"
train_file = "data/train.json"

# 读取问题域
with open(train_file, 'r', encoding='utf-8') as file:
    data = json.load(file)
infoList = []

# 遍历数据中的每个案例
for i, case in enumerate(data):
    if i >= 50:
        break
    case_id = case["案例编号"]
    clinical_info = case["临床资料"]
    mechanism_options = case["病机选项"]
    syndrome_options = case["证候选项"]
    infoList.append({"案例编号": case_id, "临床资料": clinical_info, "病机选项": mechanism_options,
                     "证候选项": syndrome_options})

# 读取知识库
index1, nodes1 = rag.load_data(["data/evalTrainTask1.json"], "evalstore01")
index2, nodes2 = rag.load_data(["data/evalTrainTask2.json"], "evalstore02")

# 生成答案
solution.task1Solution(infoList, with_LLMrerank, rerank_top_k, hybrid_mode, index1, top_k, hybrid_search,
                       nodes1, with_hyde, submitPath)
resnum = 0
lines = []
if os.path.exists(submitPath + "-1.txt"):
    with open(submitPath + "-1.txt", 'r', encoding='utf-8', errors="ignore") as file:
        lines = file.readlines()
resultList = []

qa_core_mechanism_prompt_tmpl_str = """
你是一名中医专家，请以如下案例的方式根据抽取到的核心临床信息，完成病机推理。
你的推理过程应该是：1.根据[核心临床信息]进行[病机推断]；2.根据[病机推断]筛选出[核心病机]；3.筛选出最有可能正确的数个[核心病机]；4.根据[核心病机]在[病机选项]中选出对应的[病机答案]。
注意：筛选出的正确选项最多选三个。要求：答案只包含正确选项，不含任何误导性。
---------------------
{context_str}
---------------------
[临床资料]: {clinical_info}
[核心临床信息]: {core_clinical_info}
[病机选项]: {mechanism_options}
[病机推断]: 
[核心病机]: 
[病机答案]: 
"""

for i, info in tqdm(enumerate(infoList), total=len(infoList)):
    case_id = info["案例编号"]
    clinical_info = info["临床资料"]
    mechanism_options = info["病机选项"]
    core_clinical_info = lines[i].split("@")[1].rstrip("\n")
    if len(lines[i].split("@")) > 2:
        resnum += 1
        continue
    qa_core_mechanism_prompt_tmpl_str_temp = qa_core_mechanism_prompt_tmpl_str.format(context_str="{context_str}",
                                                                                      clinical_info=clinical_info,
                                                                                      core_clinical_info=core_clinical_info,
                                                                                      mechanism_options=mechanism_options)
    prompt2 = PromptTemplate(qa_core_mechanism_prompt_tmpl_str_temp)
    rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index2, top_k,
                                              hybrid_search, nodes2, with_hyde, qa_prompt_tmpl=prompt2)
    response = rag_query_engine.query(core_clinical_info)
    mechanism_answer = str(response)
    tools.printf(f"mechanism_answer:{mechanism_answer}")
    mechanism_answer = tools.select_answers_parse(mechanism_answer, "病机答案")
    tools.printf(mechanism_answer)
    mechanism_str = tools.extract_core_mechanism(mechanism_options, mechanism_answer)
    tools.printf(mechanism_str)
    mechanism_answer_str = ""
    for ma in mechanism_answer:
        mechanism_answer_str += ma + ';'
    mechanism_answer_str = mechanism_answer_str[:-1]
    resultList.append(f"{case_id}@{core_clinical_info}@{mechanism_answer_str}")
if resnum != len(infoList):
    with open(submitPath + "-2+.txt", "w", encoding="utf-8") as file:
        for item in resultList:
            file.write(item + '\n')

# 读取生成的答案
result1 = load_all_evalData(submitPath + "-2+.txt", 1)
result2 = load_all_evalData(submitPath + "-2+.txt", 2)

# 计算结果
result = evalTask1(result1, train_file)
resultp = evalTask2(result2, train_file)
