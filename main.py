import time

from dotenv import load_dotenv, find_dotenv
from dotenv import dotenv_values
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core import Settings
from custom.glmfz import ChatGLM
from langchain.prompts import ChatPromptTemplate
from custom.prompt import clinical_info_prompt_tmpl_str, core_mechanism_prompt_tmpl_str, syndrome_infer_prompt_tmpl_str, \
    clinical_exp_prompt_tmpl_str, qa_clinical_info_prompt_tmpl_str, qa_core_mechanism_prompt_tmpl_str, \
    qa_syndrome_infer_prompt_tmpl_str, qa_clinical_exp_prompt_tmpl_str
import re
from llama_index.core import PromptTemplate
from pipeline import rag
import pipeline.readData as readData
import tools
from tqdm import *

if __name__ == '__main__':
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
        model="glm-4",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        is_chat_model=True,
    )
    # 设置参数
    with_hyde = False  # 是否采用假设文档
    persist_dir = "storeQ"  # 向量存储地址
    hybrid_search = False  # 是否采用混合检索
    top_k = 3
    response_mode = ResponseMode.SIMPLE_SUMMARIZE  # 最佳实践为为TREE_SUMMARIZE
    submitPath = "submit/sub03.txt"
    A_file = "data/A榜.json"
    train_file = "data/train.json"
    infoList = readData.read_AJson(A_file)
    index1, nodes1 = rag.load_data("data/trainTask1.json", "store01")
    index2, nodes2 = rag.load_data("data/trainTask2.json", "store02")
    index3, nodes3 = rag.load_data("data/trainTask3.json", "store03")
    index4, nodes4 = rag.load_data("data/trainTask4.json", "store04")
    subResult = []
    start_time = time.time()
    print(start_time)
    for info in tqdm(infoList):
        case_id = info["案例编号"]
        clinical_info = info["临床资料"]
        mechanism_options = info["病机选项"]
        syndrome_options = info["证候选项"]
        # task1

        prompt1 = PromptTemplate(qa_clinical_info_prompt_tmpl_str).format(context_str="{context_str}",
                                                                          clinical_info=clinical_info)
        rag_query_engine = rag.Build_query_engine(index1, top_k, hybrid_search, nodes1, with_hyde,
                                                  qa_prompt_tmpl=prompt1)
        response = rag_query_engine.query(clinical_info)
        snodes = response.source_nodes
        context_strs = ""
        for node in snodes:
            context_strs += node.text + '/n'
        prompt1 = PromptTemplate(qa_clinical_info_prompt_tmpl_str).format(context_str=context_strs,
                                                                          clinical_info=clinical_info)
        response = Settings.llm.complete(prompt1)
        core_clinical_info = str(response)
        # tools.printf(f"core_clinical_info:{core_clinical_info}")

        # task2
        prompt2 = PromptTemplate(qa_core_mechanism_prompt_tmpl_str).format(context_str="{context_str}",
                                                                           clinical_info=clinical_info,
                                                                           core_clinical_info=core_clinical_info,
                                                                           mechanism_options=mechanism_options)
        rag_query_engine = rag.Build_query_engine(index2, top_k, hybrid_search, nodes2, with_hyde,
                                                  qa_prompt_tmpl=prompt1)
        response = rag_query_engine.query(core_clinical_info)
        snodes = response.source_nodes
        context_strs = ""
        for node in snodes:
            context_strs += node.text + '/n'
        prompt2 = PromptTemplate(qa_core_mechanism_prompt_tmpl_str).format(context_str=context_strs,
                                                                           clinical_info=clinical_info,
                                                                           core_clinical_info=core_clinical_info,
                                                                           mechanism_options=mechanism_options)
        response = Settings.llm.complete(prompt2)
        mechanism_answer = str(response)
        # tools.printf(f"mechanism_answer:{mechanism_answer}")
        mechanism_answer = tools.select_answers_parse(mechanism_answer, "[病机答案]:")
        # tools.printf(mechanism_answer)
        mechanism_str = tools.extract_core_mechanism(mechanism_options, mechanism_answer)
        # tools.printf(mechanism_str)

        # task3

        prompt3 = PromptTemplate(qa_syndrome_infer_prompt_tmpl_str).format(context_str="{context_str}",
                                                                           core_clinical_info=core_clinical_info,
                                                                           syndrome_options=syndrome_options,
                                                                           mechanism_str=mechanism_str)
        rag_query_engine = rag.Build_query_engine(index3, top_k, hybrid_search, nodes3, with_hyde,
                                                  qa_prompt_tmpl=prompt3)
        response = rag_query_engine.query(mechanism_str)
        snodes = response.source_nodes
        context_strs = ""
        for node in snodes:
            context_strs += node.text + '/n'
        prompt3 = PromptTemplate(qa_syndrome_infer_prompt_tmpl_str).format(context_str=context_strs,
                                                                           core_clinical_info=core_clinical_info,
                                                                           syndrome_options=syndrome_options,
                                                                           mechanism_str=mechanism_str)
        response = Settings.llm.complete(prompt3)
        syndrome_answer = str(response)
        # tools.printf(f"syndrome_answer:{syndrome_answer}")
        syndrome_answer = tools.select_answers_parse(syndrome_answer, "[证候答案]:")
        # tools.printf(syndrome_answer)
        syndrome_answer_str = tools.extract_core_mechanism(syndrome_options, syndrome_answer)
        # tools.printf(syndrome_answer_str)

        # task4
        prompt4 = PromptTemplate(qa_clinical_exp_prompt_tmpl_str).format(context_str="{context_str}",
                                                                         clinical_info=clinical_info,
                                                                         mechanism_str=mechanism_str)
        rag_query_engine = rag.Build_query_engine(index4, top_k, hybrid_search, nodes4, with_hyde,
                                                  qa_prompt_tmpl=prompt4)
        response = rag_query_engine.query(mechanism_str)
        snodes = response.source_nodes
        context_strs = ""
        for node in snodes:
            context_strs += node.text + '/n'
        prompt4 = PromptTemplate(qa_clinical_exp_prompt_tmpl_str).format(context_str=context_strs,
                                                                         clinical_info=clinical_info,
                                                                         mechanism_str=mechanism_str)
        response = Settings.llm.complete(prompt4)
        clinical_experience_str = str(response)
        # tools.printf(f"clinical_experience_str:{clinical_experience_str}")
        diagnosis = syndrome_answer_str.split(";")
        diagnosis_str = ""
        for d in diagnosis:
            if d:
                diagnosis_str += d + '，'
        diagnosis_str = diagnosis_str[:-1]
        # tools.printf(diagnosis_str)

        resultAll = tools.printInfo(case_id, core_clinical_info, mechanism_answer, syndrome_answer,
                                    clinical_experience_str,
                                    diagnosis_str)
        subResult.append(resultAll)
    end_time = time.time()
    print(f"start_time-end_time")
    tools.saveTxt(submitPath, subResult)
