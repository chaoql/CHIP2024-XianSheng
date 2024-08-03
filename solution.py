from custom.prompt import qa_clinical_info_prompt_tmpl_str, qa_core_mechanism_prompt_tmpl_str, \
    qa_syndrome_infer_prompt_tmpl_str, qa_clinical_exp_prompt_tmpl_str
from llama_index.core import PromptTemplate
from pipeline import rag
import tools
from tqdm import *
from tools import saveTxt, printf
import os


def task1Solution(infoList, resnum, with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k, hybrid_search, nodes,
                  with_hyde, submitPath):
    resultList = []
    for i, info in tqdm(enumerate(infoList), total=len(infoList)):
        if i < resnum:
            continue
        case_id = info["案例编号"]
        clinical_info = info["临床资料"]
        qa_clinical_info_prompt_tmpl_str_temp = qa_clinical_info_prompt_tmpl_str.format(context_str="{context_str}",
                                                                                        clinical_info=clinical_info)
        prompt1 = PromptTemplate(qa_clinical_info_prompt_tmpl_str_temp)
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k,
                                                  hybrid_search, nodes, with_hyde, qa_prompt_tmpl=prompt1)
        response = rag_query_engine.query(clinical_info)
        core_clinical_info = str(response)
        printf(f"core_clinical_info:{core_clinical_info}")
        resultList.append(f"{case_id}@{core_clinical_info}")
    with open(submitPath + "-1.txt", "w", encoding="utf-8") as file:
        for item in resultList:
            file.write(item)


def task2Solution(infoList, resnum, with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k, hybrid_search, nodes,
                  with_hyde, submitPath):
    resnum = 0
    lines = []
    if os.path.exists(submitPath + "-1.txt"):
        with open(submitPath + "-1.txt", 'r', encoding='utf-8') as file:
            lines = file.readlines()
            resnum = len(lines)
    resultList = []
    for i, info in tqdm(enumerate(infoList), total=len(infoList)):
        # if i < resnum:
        #     continue
        case_id = info["案例编号"]
        clinical_info = info["临床资料"]
        mechanism_options = info["病机选项"]
        core_clinical_info = lines[i].split("@")[1]
        qa_core_mechanism_prompt_tmpl_str_temp = qa_core_mechanism_prompt_tmpl_str.format(context_str="{context_str}",
                                                                                          clinical_info=clinical_info,
                                                                                          core_clinical_info=core_clinical_info,
                                                                                          mechanism_options=mechanism_options)
        prompt2 = PromptTemplate(qa_core_mechanism_prompt_tmpl_str_temp)
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k,
                                                  hybrid_search, nodes, with_hyde, qa_prompt_tmpl=prompt2)
        response = rag_query_engine.query(core_clinical_info)
        mechanism_answer = str(response)
        tools.printf(f"mechanism_answer:{mechanism_answer}")
        mechanism_answer = tools.select_answers_parse(mechanism_answer, "[病机答案]:")
        tools.printf(mechanism_answer)
        mechanism_str = tools.extract_core_mechanism(mechanism_options, mechanism_answer)
        tools.printf(mechanism_str)
        mechanism_answer_str = ""
        for ma in mechanism_answer:
            mechanism_answer_str += ma + ';'
        mechanism_answer_str = mechanism_answer_str[:-1]
        resultList.append(f"{case_id}@{core_clinical_info}@{mechanism_answer_str}")
    with open(submitPath + "-2.txt", "w", encoding="utf-8") as file:
        for item in resultList:
            file.write(item)


def task3Solution(infoList, resnum, with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k, hybrid_search, nodes,
                  with_hyde, submitPath):
    resnum = 0
    lines = []
    if os.path.exists(submitPath + "-2.txt"):
        with open(submitPath + "-2.txt", 'r', encoding='utf-8') as file:
            lines = file.readlines()
            resnum = len(lines)
    resultList = []
    for i, info in tqdm(enumerate(infoList), total=len(infoList)):
        # if i < resnum:
        #     continue
        case_id = info["案例编号"]
        mechanism_options = info["病机选项"]
        syndrome_options = info["证候选项"]
        core_clinical_info = lines[i].split("@")[1]
        mechanism_answer_str = lines[i].split("@")[2]
        mechanism_answer = mechanism_answer_str.split(";")
        mechanism_str = tools.extract_core_mechanism(mechanism_options, mechanism_answer)
        qa_syndrome_infer_prompt_tmpl_str_temp = qa_syndrome_infer_prompt_tmpl_str.format(context_str="{context_str}",
                                                                                          core_clinical_info=core_clinical_info,
                                                                                          syndrome_options=syndrome_options,
                                                                                          mechanism_str=mechanism_str)
        prompt3 = PromptTemplate(qa_syndrome_infer_prompt_tmpl_str_temp)
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k,
                                                  hybrid_search, nodes, with_hyde, qa_prompt_tmpl=prompt3)
        response = rag_query_engine.query(mechanism_str)
        syndrome_answer = str(response)
        tools.printf(f"syndrome_answer:{syndrome_answer}")
        syndrome_answer = tools.select_answers_parse(syndrome_answer, "[证候答案]:")
        tools.printf(syndrome_answer)
        syndrome_answer_str = tools.extract_core_mechanism(syndrome_options, syndrome_answer)
        tools.printf(syndrome_answer_str)
        syndrome_answer_str = ""
        for sa in syndrome_answer:
            syndrome_answer_str += sa + ';'
        syndrome_answer_str = syndrome_answer_str[:-1]
        resultList.append(f"{case_id}@{core_clinical_info}@{mechanism_answer_str}@{syndrome_answer_str}")
    with open(submitPath + "-3.txt", "w", encoding="utf-8") as file:
        for item in resultList:
            file.write(item)


def task4Solution(infoList, resnum, with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k, hybrid_search, nodes,
                  with_hyde, submitPath):
    resnum = 0
    lines = []
    if os.path.exists(submitPath + "-3.txt"):
        with open(submitPath + "-3.txt", 'r', encoding='utf-8') as file:
            lines = file.readlines()
            resnum = len(lines)
    resultList = []
    for i, info in tqdm(enumerate(infoList), total=len(infoList)):
        # if i < resnum:
        #     continue
        case_id = info["案例编号"]
        clinical_info = info["临床资料"]
        mechanism_options = info["病机选项"]
        syndrome_options = info["证候选项"]
        linsI = lines[i].split("@")
        core_clinical_info = linsI[1]
        mechanism_answer_str = linsI[2]
        mechanism_answer = mechanism_answer_str.split(";")
        mechanism_str = tools.extract_core_mechanism(mechanism_options, mechanism_answer)
        syndrome_answer_str = linsI[3]
        syndrome_answer = syndrome_answer_str.split(";")
        syndrome_str = tools.extract_core_mechanism(syndrome_options, syndrome_answer)

        qa_clinical_exp_prompt_tmpl_str_temp = qa_clinical_exp_prompt_tmpl_str.format(context_str="{context_str}",
                                                                                      clinical_info=clinical_info,
                                                                                      mechanism_str=mechanism_str,
                                                                                      syndrome_answer_str=syndrome_str, )
        prompt4 = PromptTemplate(qa_clinical_exp_prompt_tmpl_str_temp)
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k,
                                                  hybrid_search, nodes, with_hyde, qa_prompt_tmpl=prompt4)
        response = rag_query_engine.query(mechanism_str)
        clinical_experience_str = str(response)
        tools.printf(f"clinical_experience_str:{clinical_experience_str}")
        diagnosis = syndrome_str.split(";")
        diagnosis_str = ""
        for d in diagnosis:
            if d:
                diagnosis_str += d + '，'
        diagnosis_str = diagnosis_str[:-1]
        tools.printf(diagnosis_str)
        resultList.append(
            f"{case_id}@{core_clinical_info}@{mechanism_answer_str}@{syndrome_answer_str}@临证体会：{clinical_experience_str}辨证：{diagnosis_str}")
    with open(submitPath + "-4.txt", "w", encoding="utf-8") as file:
        for item in resultList:
            file.write(item)
