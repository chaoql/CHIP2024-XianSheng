import time
from dotenv import load_dotenv, find_dotenv
from dotenv import dotenv_values
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core import Settings
from custom.glmfz import ChatGLM
from custom.prompt import qa_clinical_info_prompt_tmpl_str, qa_core_mechanism_prompt_tmpl_str, \
    qa_syndrome_infer_prompt_tmpl_str, qa_clinical_exp_prompt_tmpl_str, new_qa_core_mechanism_prompt_tmpl_str, \
    all_clinical_info_prompt_tmpl_str
from llama_index.core import PromptTemplate
from pipeline import rag
import pipeline.readData as readData
import tools
import ensemble
from tqdm import *
import os
import warnings
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

warnings.filterwarnings('ignore')
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
    # Settings.embed_model = HuggingFaceEmbedding(
    #     model_name="lier007/xiaobu-embedding-v2",
    #     cache_folder="BAAI/",
    #     embed_batch_size=128,
    #     local_files_only=True,  # 仅加载本地模型，不尝试下载
    #     device="cuda",
    # )

    # 加载大模型
    # Settings.llm = DashScope(
    #     model_name=DashScopeGenerationModels.QWEN_TURBO, api_key=config["DASHSCOPE_API_KEY"], max_tokens=1024
    # )
    Settings.llm = ChatGLM(
        api_key=config["GLM_KEY"],
        model="glm-4-plus",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        is_chat_model=True,
    )

    # 设置参数
    with_hyde = False  # 是否采用假设文档
    persist_dir = "storeQ"  # 向量存储地址
    hybrid_search = False  # 是否采用混合检索
    with_LLMrerank = True
    top_k = 5
    rerank_top_k = 2
    r_nums = 10  # 循环次数
    response_mode = ResponseMode.TREE_SUMMARIZE
    submitPath = "final_submit/sub_5.txt"
    now_best_submitPath = "final_submit/sub_8.txt"
    A_file = "rdata/B榜.json"
    train_file = "rdata/train.json"
    infoList = readData.read_AJson(A_file)
    if hybrid_search:
        index1, nodes1 = rag.load_hybrid_data(["rdata/trainTask1.json"], "final_store/hybrid_store01")
        # index2, nodes2 = rag.load_data(["data/bjtrain02ExtraData.json"], "store/eval02_2_extra")
        index2, nodes2 = rag.load_hybrid_data(["rdata/trainTask2WOoptions2.json"], "final_store/hybrid_store02_1")
        index2R, nodes2R = rag.load_hybrid_data(["rdata/trainTask2WOoptions3.json"], "final_store/hybrid_store02_2")
        index3, nodes3 = rag.load_hybrid_data(
            ["rdata/zntrain3.json", "rdata/trainTask3WOoptions.json", "rdata/trainExtraDataTask2.json",
             "data/bjtrain02ExtraData.json"], "final_store/hybrid_store03_2")
        index4, nodes4 = rag.load_json_text_hybrid_data(
            ["rdata/trainTask4.json"], ["rdata/中医诊断学2.txt", "rdata/中医内科学.txt", "rdata/中医基础理论.txt"],
            "final_store/hybrid_store04")
    else:
        index1, nodes1 = rag.load_data(["rdata/trainTask1.json"], "final_store/store01")
        index2, nodes2 = rag.load_data(["rdata/trainTask2WOoptions2.json"], "final_store/store02_1")
        index2R, nodes2R = rag.load_data(["rdata/trainTask2WOoptions3.json"], "final_store/store02_2")
        index3, nodes3 = rag.load_data(
            ["rdata/zntrain3.json", "rdata/trainTask3WOoptions.json", "rdata/trainExtraDataTask2.json",
             "rdata/bjtrain02ExtraData.json"], "final_store/store03_2")
        index4, nodes4 = rag.load_json_text_data(
            ["rdata/trainTask4.json"], ["rdata/中医诊断学2.txt", "rdata/中医内科学.txt", "rdata/中医基础理论.txt"],
            "final_store/store04")

    subResult = []
    start_time = time.time()
    print(start_time)
    resnum = 0
    if os.path.exists(submitPath):
        with open(submitPath, 'r', encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()
            resnum = len(lines)

    # 读取历史最佳答案，某个病例生成错误后直接替换为历史最佳答案，防止报错打断程序运行
    # 主要是因为LLM_reranker报错率太高了
    task1_ans = []
    task2_ans = []
    task3_ans = []
    task4_ans = []
    with open(now_best_submitPath, 'r', encoding="utf-8", errors="ignore") as file:
        temp = file.readlines()
    for i in temp:
        task1_ans.append(i.split("@")[1].rstrip())
        task2_ans.append(i.split("@")[2].rstrip())
        task3_ans.append(i.split("@")[3].rstrip())
        task4_ans.append(i.split("@")[4].rstrip())

    for i, info in tqdm(enumerate(infoList), total=len(infoList)):
        if i < resnum:  # 接续生成
            continue

        global syndrome_answer_str, mechanism_str, syndrome_answer, mechanism_answer
        case_id = info["案例编号"]
        clinical_info = info["临床资料"]
        mechanism_options = info["病机选项"]
        syndrome_options = info["证候选项"]

        # 直接读取各题当前最佳答案，调试用
        # core_clinical_info_str = task1_ans[i]
        # tools.printf(f"core_clinical_info_str:{core_clinical_info_str}")
        #
        # mechanism_answer = task2_ans[i].split(";")
        # mechanism_str = tools.extract_core_mechanism(mechanism_options, mechanism_answer)
        # tools.printf(mechanism_str)
        #
        # syndrome_answer = task3_ans[i].split(";")
        # syndrome_answer_str = tools.extract_core_mechanism(syndrome_options, syndrome_answer)
        # tools.printf(syndrome_answer_str)
        #
        # task4 = task4_ans[i]
        # clinical_experience_str = task4.split("辨证")[0].replace("临证体会：", "")
        # diagnosis_str = task4.split("辨证")[1].replace("辨证：", "").replace("：", "")

        # task1
        qa_clinical_info_prompt_tmpl_str_temp = qa_clinical_info_prompt_tmpl_str.format(context_str="{context_str}",
                                                                                        clinical_info=clinical_info)
        prompt1 = PromptTemplate(qa_clinical_info_prompt_tmpl_str_temp)
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, index1, top_k, response_mode,
                                                  hybrid_search, nodes1, with_hyde, qa_prompt_tmpl=prompt1)
        response = rag_query_engine.query(clinical_info)

        core_clinical_info = str(response)
        tools.printf(f"core_clinical_info:{core_clinical_info}")

        all_clinical_info_prompt_tmpl_str_temp = all_clinical_info_prompt_tmpl_str.format(clinical_info=clinical_info)
        prompt1_1 = PromptTemplate(all_clinical_info_prompt_tmpl_str_temp)
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, index1, top_k, response_mode,
                                                  hybrid_search, nodes1, with_hyde, qa_prompt_tmpl=prompt1_1)
        response1_1 = rag_query_engine.query(clinical_info)
        core_clinical_info1_1 = str(response1_1)
        core_clinical_info_list = core_clinical_info.split(";")
        print(len(core_clinical_info_list))

        core_clinical_info1_1_list = core_clinical_info1_1.split(";")
        core_clinical_info_result = []
        core_clinical_info_str = ""
        for info in core_clinical_info_list:
            if info:
                core_clinical_info1_1_list.append(info)
        for info in core_clinical_info1_1_list:
            if info and info not in core_clinical_info_result:
                core_clinical_info_result.append(info)
                core_clinical_info_str += info + ";"
        print(len(core_clinical_info_result))

        for j in range(r_nums):
            # task2
            try:
                if j == 0:
                    qa_core_mechanism_prompt_tmpl_str_temp = qa_core_mechanism_prompt_tmpl_str.format(
                        context_str="{context_str}",
                        core_clinical_info=core_clinical_info,
                        mechanism_options=mechanism_options,
                    )
                    prompt2 = PromptTemplate(qa_core_mechanism_prompt_tmpl_str_temp)
                    rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, index2, top_k,
                                                              response_mode, hybrid_search, nodes2, with_hyde,
                                                              qa_prompt_tmpl=prompt2)
                    response = rag_query_engine.query(core_clinical_info)  #
                else:
                    syndrome_answer_str = tools.extract_core_mechanism(syndrome_options, syndrome_answer)
                    qa_core_mechanism_prompt_tmpl_str_temp = new_qa_core_mechanism_prompt_tmpl_str.format(
                        context_str="{context_str}",
                        core_clinical_info=core_clinical_info,
                        mechanism_options=mechanism_options,
                        syndrome_answer_str=syndrome_answer_str
                    )
                    prompt2 = PromptTemplate(qa_core_mechanism_prompt_tmpl_str_temp)
                    rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, index2R, top_k,
                                                              response_mode,
                                                              hybrid_search, nodes2R, with_hyde, qa_prompt_tmpl=prompt2)
                    response = rag_query_engine.query(syndrome_answer_str)

                mechanism_answer = str(response)
                tools.printf(f"mechanism_answer:{mechanism_answer}")
                mechanism_answer = tools.select_answers_parse(mechanism_answer, "病机答案")
                tools.printf(mechanism_answer)
                mechanism_str = tools.extract_core_mechanism(mechanism_options, mechanism_answer)
                tools.printf(mechanism_str)
            except Exception as e:
                mechanism_answer = task2_ans[i].split(";")
                mechanism_str = tools.extract_core_mechanism(mechanism_options, mechanism_answer)
                tools.printf(mechanism_str)

            # task3
            qa_syndrome_infer_prompt_tmpl_str_temp = qa_syndrome_infer_prompt_tmpl_str.format(
                context_str="{context_str}",
                core_clinical_info=core_clinical_info,
                syndrome_options=syndrome_options,
                mechanism_str=mechanism_str)
            prompt3 = PromptTemplate(qa_syndrome_infer_prompt_tmpl_str_temp)
            rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, index3, top_k, response_mode,
                                                      hybrid_search, nodes3, with_hyde, qa_prompt_tmpl=prompt3)
            try:
                response = rag_query_engine.query(mechanism_str + core_clinical_info)
                for node in response.source_nodes:
                    tools.printf(node.text)
                syndrome_answer = str(response)
                tools.printf(f"syndrome_answer:{syndrome_answer}")
                syndrome_answer = tools.select_answers_parse(syndrome_answer, "证候答案")
                tools.printf(syndrome_answer)
                syndrome_answer_str = tools.extract_core_mechanism(syndrome_options, syndrome_answer)
                tools.printf(syndrome_answer_str)
            except Exception as e:
                syndrome_answer = task3_ans[i].split(";")
                syndrome_answer_str = tools.extract_core_mechanism(syndrome_options, syndrome_answer)
                tools.printf(syndrome_answer_str)

        # task4
        qa_clinical_exp_prompt_tmpl_str_temp = qa_clinical_exp_prompt_tmpl_str.format(context_str="{context_str}",
                                                                                      clinical_info=clinical_info,
                                                                                      mechanism_str=mechanism_str)
        prompt4 = PromptTemplate(qa_clinical_exp_prompt_tmpl_str_temp)
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, index4, top_k, response_mode,
                                                  hybrid_search, nodes4, with_hyde, qa_prompt_tmpl=prompt4)
        try:
            response = rag_query_engine.query(mechanism_str)
            clinical_experience_str = str(response).replace("\n", "")
            tools.printf(f"clinical_experience_str:{clinical_experience_str}")
            diagnosis = syndrome_answer_str.split(";")
            diagnosis_str = ""
            for d in diagnosis:
                if d:
                    diagnosis_str += d + '，'
            diagnosis_str = diagnosis_str[:-1]
            tools.printf(diagnosis_str)
        except Exception as e:
            task4 = task4_ans[i]
            clinical_experience_str = task4.split("辨证")[0].replace("临证体会：", "")
            diagnosis_str = task4.split("辨证")[1].replace("辨证：", "").replace("：", "")

        resultAll = tools.printInfo(case_id, core_clinical_info_str, mechanism_answer, syndrome_answer,
                                    clinical_experience_str, diagnosis_str)
        tools.save_now(resultAll, submitPath)
        subResult.append(resultAll)
    end_time = time.time()
    print(f"start_time-end_time:{(end_time - start_time) / 60.0}min")
    tools.saveTxt("submit/subTemp.txt", subResult)

    ensemble.enseTask1(submitPath=submitPath, llmsubmit_path=submitPath, A_path=A_file)  # 第一问补缺
