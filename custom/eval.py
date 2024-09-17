import json
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
from custom.kimi import ChatKIMI
from llama_index.llms.openai import OpenAI
from rouge_chinese import Rouge
import jieba


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
        print(f"{train_data[i]['案例编号']}: {score}")
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
        print(f"{train_data[i]['案例编号']}: {score}\t正确答案：{trainList} ||提交答案：{resList}")
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
        print(f"{train_data[i]['案例编号']}: {score}\t正确答案：{trainList} ||提交答案：{resList}")
    print(f"task3总得分: {all_score}/50")
    return all_score


def evalTask4(result, train_path):
    with open(train_path, 'r', encoding='utf-8', errors='ignore') as file:
        train_data = json.load(file)
    all_score = 0.0
    for i, res in enumerate(result):
        hypothesis = train_data[i]["临证体会"] + train_data[i]["辨证"]
        hypothesis = ' '.join(jieba.cut(hypothesis))

        reference = res
        reference = ' '.join(jieba.cut(reference))

        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        rouge_l = scores[0]["rouge-l"]["f"]
        all_score += rouge_l
        print(f"{train_data[i]['案例编号']}: {rouge_l}")
    print(f"task4总得分: {all_score}/50")
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
# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-large-zh-v1.5",
#     cache_folder="./BAAI/",
#     embed_batch_size=128,
#     local_files_only=True,  # 仅加载本地模型，不尝试下载
#     device="cuda",
# )
Settings.embed_model = HuggingFaceEmbedding(
    model_name="lier007/xiaobu-embedding-v2",
    cache_folder="BAAI/",
    embed_batch_size=128,
    local_files_only=True,  # 仅加载本地模型，不尝试下载
    device="cuda",
)
# 加载大模型
Settings.llm = ChatGLM(
    api_key=config["GLM_KEY"],
    model="GLM-4-0520",
    api_base="https://open.bigmodel.cn/api/paas/v4/",
    is_chat_model=True,
    context_window=6000,
)


# Settings.llm = OpenAI(model="gpt-4o")

# from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
#
# Settings.llm = DashScope(
#     model_name=DashScopeGenerationModels.QWEN_MAX, api_key=config["DASHSCOPE_API_KEY"], max_tokens=1024
# )

# 设置参数
with_hyde = False  # 是否采用假设文档
persist_dir = "storeQ"  # 向量存储地址
hybrid_search = False  # 是否采用混合检索
with_LLMrerank = True
top_k = 5
rerank_top_k = 2
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
    infoList.append(
        {"案例编号": case_id, "临床资料": clinical_info, "病机选项": mechanism_options, "证候选项": syndrome_options})

# 读取知识库
# index1, nodes1 = rag.load_data(["data/evalTrainTask1.json"], "store/eval01")

# from llama_index.core import PromptTemplate, get_response_synthesizer, StorageContext, VectorStoreIndex, \
#     SimpleDirectoryReader, Settings
# from llama_index.core.node_parser import JSONNodeParser
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core import load_index_from_storage
#
# top_k = 7
# rerank_top_k = 2
# documents = SimpleDirectoryReader(input_files=["data/evalTrainTask2_2.json"]).load_data()
# node_parser = JSONNodeParser()
# nodes2_1 = node_parser.get_nodes_from_documents(documents, show_progress=True)
# documents = SimpleDirectoryReader(input_dir="extra_data/病机").load_data()
# node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
# nodes2_2 = node_parser.get_nodes_from_documents(documents, show_progress=True)
# nodes2 = nodes2_1 + nodes2_2
# # indexing & storing
# try:
#     storage_context = StorageContext.from_defaults(persist_dir="store/eval02Extra_2")
#     index2 = load_index_from_storage(storage_context, show_progress=True)
# except:
#     index2 = VectorStoreIndex(nodes=nodes2, show_progress=True)
#     index2.storage_context.persist(persist_dir="store/eval02Extra_2")

# indexK2, nodesk2 = rag.load_data(["data/bjtrain02ExtraData.json"], "store/eval02_2_extra")
# index2, nodes2 = rag.load_hybrid_data(["data/bjtrain02ExtraData.json"], "store/hybrid_eval02_2_extra")

# index2, nodes2 = rag.load_data(["data/evalTrainTask2_2.json"], "store/eval02_2")
index2, nodes2 = rag.load_data(["data/evalTrainTask2_3.json"], "store/xiaobu_eval02_3")
# index2, nodes2 = rag.load_txt_data(
#     ["extra_data/doctor/中医内科学.txt", "extra_data/doctor/中医基础理论.txt", "extra_data/doctor/中医诊断学.txt"],
#     "store/eval02_doctor")


index3, nodes3 = rag.load_data(["data/evalTrainTask3.json", "data/trainExtraData.json"],
                               "store/xiaobu_eval03")

# index3, nodes3 = rag.load_data(
#     ["data/zntrain3.json", "data/evalTrainTask3.json", "data/trainExtraDataTask2.json",
#      "data/bjtrain02ExtraData.json", ], "store/zn2")

# index3, nodes3 = rag.load_data(
#     ["data/zntrain3.json", "data/evalTrainTask3.json", "data/trainExtraDataTask2.json", "data/bjtrain02ExtraData.json",
#      "data/trainExtraData04.json"], "store/zn5")
# index3, nodes3 = rag.load_data(["data/evalTrainTask3.json", "data/trainExtraDataTask2.json"],
#                                "store/zn4")
# index3, nodes3 = rag.load_data(
#     ["data/trainTask3WOoptions.json", "data/trainExtraData.json", "data/bjtrain02ExtraData.json"],
#     "store/storeExtra01")
#
from llama_index.core import PromptTemplate, get_response_synthesizer, StorageContext, VectorStoreIndex, \
    SimpleDirectoryReader, Settings
from llama_index.core.node_parser import JSONNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import load_index_from_storage

top_k = 5
rerank_top_k = 2
documents = SimpleDirectoryReader(input_files=["data/trainTask4.json"]).load_data()
node_parser = JSONNodeParser()
nodes4_1 = node_parser.get_nodes_from_documents(documents, show_progress=True)
documents = SimpleDirectoryReader(input_files=["extra_data/doctor/中医诊断学2.txt", "extra_data/doctor/中医内科学.txt",
                                               "extra_data/doctor/中医基础理论.txt", ]).load_data()
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
nodes4_2 = node_parser.get_nodes_from_documents(documents, show_progress=True)
nodes4 = nodes4_1 + nodes4_2
# indexing & storing
try:
    storage_context = StorageContext.from_defaults(persist_dir="store/xiaohu_store04_4")
    index4 = load_index_from_storage(storage_context, show_progress=True)
except:
    index4 = VectorStoreIndex(nodes=nodes4, show_progress=True)
    index4.storage_context.persist(persist_dir="store/xiaohu_store04_4")
# index4, nodes4 = rag.load_data(["data/trainTask4_2.json"], "store/store04")

# 生成答案
# solution.task1Solution(infoList, False, rerank_top_k, hybrid_mode, index1, top_k, hybrid_search,
#                        nodes1, with_hyde, submitPath)
# solution.task2Solution(infoList, with_LLMrerank, rerank_top_k, hybrid_mode, index2, top_k, hybrid_search,
#                        nodes2, with_hyde, submitPath)
# solution.task3Solution(infoList, with_LLMrerank, rerank_top_k, hybrid_mode, index3, top_k, hybrid_search,
#                        nodes3, with_hyde, submitPath)
solution.task4Solution(infoList, with_LLMrerank, rerank_top_k, hybrid_mode, index4, top_k, hybrid_search,
                       nodes4, with_hyde, submitPath)

# 读取生成的答案
# result1 = load_all_evalData(submitPath + "-1.txt", 1)
# result2 = load_all_evalData(submitPath + "-2.txt", 2)
# result3 = load_all_evalData(submitPath + "-3.txt", 3)
result4 = load_all_evalData(submitPath + "-4.txt", 4)

# 计算结果
# resultT1 = evalTask1(result1, train_file)
# resultT2 = evalTask2(result2, train_file)
# resultT3 = evalTask3(result3, train_file)
resultT4 = evalTask4(result4, train_file)
