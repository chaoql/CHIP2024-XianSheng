import time
from dotenv import load_dotenv, find_dotenv
from dotenv import dotenv_values
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core import Settings
from custom.glmfz import ChatGLM
from custom.prompt import qa_clinical_info_prompt_tmpl_str, qa_core_mechanism_prompt_tmpl_str, \
    qa_syndrome_infer_prompt_tmpl_str, qa_clinical_exp_prompt_tmpl_str
from llama_index.core import PromptTemplate
from pipeline import rag
from custom.embedding import InstructorEmbeddings
import pipeline.readData as readData
import tools
import ensemble
from tqdm import *
import os
import warnings
from sentence_transformers import SentenceTransformer
# import solution
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from modelscope import snapshot_download

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
    # model_dir = snapshot_download("iic/gte_Qwen2-7B-instruct")
    # Settings.embed_model = InstructorEmbeddings(SentenceTransformer(model_dir, trust_remote_code=True), "")
    # 加载大模型
    # Settings.llm = DashScope(
    #     model_name=DashScopeGenerationModels.QWEN_TURBO, api_key=config["DASHSCOPE_API_KEY"], max_tokens=1024
    # )
    Settings.llm = ChatGLM(
        api_key=config["GLM_KEY"],
        model="glm-4-0520",
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
    r_nums = 3  # 循环次数
    response_mode = ResponseMode.TREE_SUMMARIZE  # 最佳实践为为TREE_SUMMARIZE
    hybrid_mode = "OR"
    submitPath = "submit/sub49.txt"
    A_file = "data/A榜.json"
    train_file = "data/train.json"
    infoList = readData.read_AJson(A_file)
    qa_core_mechanism_prompt_tmpl_str = """你是一名中医专家，请使用以下检索到的上下文而不是先验知识，来根据抽取到的核心临床信息，完成病机推理。
病机，也称病理，是中医的常见概念，指疾病的病因、病性、病位及病程中变化的要理。

对于当前问题，你的推理过程应该是：
1.根据[核心临床信息]对[病机选项]进行逐一排除；
2.统计符合患者临床信息的[病机选项]；

注意：答案不超过三个，并将最终正确的选项输出在最后一行“[病机答案]:”之后。

---------------------
{context_str}
---------------------

[核心临床信息]: {core_clinical_info}
[病机选项]: {mechanism_options}
[病机推断]:
[核心病机]:
[病机答案]:
"""
    new_qa_core_mechanism_prompt_tmpl_str = """你是一名中医专家，请使用上下文信息，根据抽取到的核心临床信息，完成病机推理。
    
证候和病机之间的关系：
证候和病机二者有密切的关系。但严格说来，证候和病机的概念不同，证候是指疾病发展阶段中的病因、病位、病性、病机、病势及邪正斗争强弱等方面情况的病理概括。而病机则是人体在一定条件下，由致病因素引起的一种以正邪相争为基本形式的病理过程。一个病机可以有不同的证候，同样相同的证候亦可见于不同的病机中，所以有“同病异证”、“异病同证”的说法。如感冒病，其证候有风寒证和风热证的不同，须用不同的治法；再如头痛与眩晕虽属两病但均可出现血虚证候。因此，既要辩证，又要辨病。辨别病机要按照辨别证候所得，与多种相类似的疾病进行鉴别比较，同时进一步指导辨证，最后把那些类似的疾病一一排除，得出疾病的结论。在得出结论之后，对该病今后病机演变已有一个梗概，在这个基础上进一步辨证，便能预料其顺逆吉凶。

你的推理过程应该是：
1.根据[核心临床信息]和[核心证候]进行[病机推断]，只选择确定可以推断出的病机，不确定的病机不予考虑；
2.根据[病机推断]筛选出[核心病机]；
3.统计[核心病机]的数量；
4.只留下最可能正确的核心病机，数量小于等于3个；
5.根据[核心病机]在[病机选项]中选出对应的[病机答案]。将最终正确的选项输出在最后一行“[病机答案]:”之后。  

---------------------
{context_str}
---------------------

[核心临床信息]: {core_clinical_info}
[核心证候]: {syndrome_answer_str}
[病机选项]: {mechanism_options}
[病机推断]: 
[核心病机]: 
[病机答案]: 
"""
    if hybrid_search:
        index1, nodes1 = rag.load_hybrid_data(["data/trainTask1.json"], "store/hybrid_store01")
        index2, nodes2 = rag.load_hybrid_data(["data/trainTask2WOoptions.json"], "store/hybrid_store02")
        index3, nodes3 = rag.load_hybrid_data(["data/trainTask3WOoptions.json", "data/trainExtraData.json"],
                                              "store/hybrid_store03")
        index4, nodes4 = rag.load_hybrid_data(["data/trainTask4.json"], "store/hybrid_store04")
    else:
        index1, nodes1 = rag.load_data(["data/trainTask1.json"], "store/store01")
        # index2, nodes2 = rag.load_data(["data/bjtrain02ExtraData.json"], "store/eval02_2_extra")
        index2, nodes2 = rag.load_data(["data/trainTask2WOoptions2.json"], "store/store02_2")
        index2R, nodes2R = rag.load_data(["data/trainTask2WOoptions3.json"], "store/store02_3")

        index3, nodes3 = rag.load_data(["data/trainTask3WOoptions.json", "data/trainExtraData.json"],
                                       "store/storeExtra")
        index4, nodes4 = rag.load_data(["data/trainTask4.json"], "store/store04")
    subResult = []
    start_time = time.time()
    print(start_time)
    resnum = 0
    if os.path.exists(submitPath):
        with open(submitPath, 'r', encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()
            resnum = len(lines)

    # task1_ans = []
    # task2_ans = []
    # task3_ans = []
    # task4_ans = []
    # with open("submit/sub42.txt", 'r', encoding="utf-8", errors="ignore") as file:
    #     temp = file.readlines()
    # for i in temp:
    #     task1_ans.append(i.split("@")[1].rstrip())
    #     task2_ans.append(i.split("@")[2].rstrip())
    #     task3_ans.append(i.split("@")[3].rstrip())
    #     task4_ans.append(i.split("@")[4].rstrip())

    for i, info in tqdm(enumerate(infoList), total=len(infoList)):
        if i < resnum:
            continue
        case_id = info["案例编号"]
        clinical_info = info["临床资料"]
        mechanism_options = info["病机选项"]
        syndrome_options = info["证候选项"]
        # task1
        qa_clinical_info_prompt_tmpl_str = """你是一名中医专家，请按照检索到的案例的格式回答当前临床资料中包含的[核心临床信息]。
要求[核心临床信息]必须包含病人的症状、舌、脉等临床信息实体。
答案中只列出以”;“分隔的核心信息，不含任何换行符等无关字符。

--------------
{context_str}
--------------

[临床资料]: {clinical_info}
[核心临床信息]: 
"""
        qa_clinical_info_prompt_tmpl_str_temp = qa_clinical_info_prompt_tmpl_str.format(context_str="{context_str}",
                                                                                        clinical_info=clinical_info)
        prompt1 = PromptTemplate(qa_clinical_info_prompt_tmpl_str_temp)
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index1, top_k,
                                                  hybrid_search, nodes1, with_hyde, qa_prompt_tmpl=prompt1)
        response = rag_query_engine.query(clinical_info)

        core_clinical_info = str(response)
        tools.printf(f"core_clinical_info:{core_clinical_info}")
        # core_clinical_info = task1_ans[i]
        # tools.printf(f"core_clinical_info:{core_clinical_info}")
        all_clinical_info_prompt_tmpl_str = """请抽取如下临床资料中与临床相关的所有实体，不包含姓名。答案只列出以”;“分隔的实体，不含任何换行符等无关字符。
[临床资料]: {clinical_info}
"""
        all_clinical_info_prompt_tmpl_str_temp = all_clinical_info_prompt_tmpl_str.format(clinical_info=clinical_info)
        prompt1_1 = PromptTemplate(all_clinical_info_prompt_tmpl_str_temp)
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index1, top_k,
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
        # core_clinical_info_str = task1_ans[i]
        # tools.printf(f"core_clinical_info_str:{core_clinical_info_str}")
        # core_clinical_info = task1_ans[i]
        # tools.printf(f"core_clinical_info:{core_clinical_info}")

        global syndrome_answer_str
        global syndrome_answer

        for i in range(r_nums):
            # task2
            if i == 0:
                qa_core_mechanism_prompt_tmpl_str_temp = qa_core_mechanism_prompt_tmpl_str.format(
                    context_str="{context_str}",
                    core_clinical_info=core_clinical_info_str,
                    mechanism_options=mechanism_options,
                )
                prompt2 = PromptTemplate(qa_core_mechanism_prompt_tmpl_str_temp)
                rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index2, top_k,
                                                          hybrid_search, nodes2, with_hyde, qa_prompt_tmpl=prompt2)
                response = rag_query_engine.query(core_clinical_info_str)  #
            else:
                syndrome_answer_str = tools.extract_core_mechanism(syndrome_options, syndrome_answer)
                qa_core_mechanism_prompt_tmpl_str_temp = new_qa_core_mechanism_prompt_tmpl_str.format(
                    context_str="{context_str}",
                    core_clinical_info=core_clinical_info_str,
                    mechanism_options=mechanism_options,
                    syndrome_answer_str=syndrome_answer_str
                )
                prompt2 = PromptTemplate(qa_core_mechanism_prompt_tmpl_str_temp)
                rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index2R, top_k,
                                                          hybrid_search, nodes2R, with_hyde, qa_prompt_tmpl=prompt2)
                response = rag_query_engine.query(syndrome_answer_str)

            mechanism_answer = str(response)
            tools.printf(f"mechanism_answer:{mechanism_answer}")
            mechanism_answer = tools.select_answers_parse(mechanism_answer, "病机答案")
            tools.printf(mechanism_answer)
            mechanism_str = tools.extract_core_mechanism(mechanism_options, mechanism_answer)
            tools.printf(mechanism_str)
            # mechanism_answer = task2_ans[i].split(";")
            # mechanism_str = tools.extract_core_mechanism(mechanism_options, mechanism_answer)
            # tools.printf(mechanism_str)

            # task3
            qa_syndrome_infer_prompt_tmpl_str = """
你是一名中医专家，请使用检索到的症状相似的其他患者的历史病例信息，根据患者[核心临床信息]和[核心病机]完成[证候推断]。

这是一道选择题，你的推理过程应该是：
1.根据[历史病例]、患者的[核心临床信息]和[核心病机]进行[证候推断]，每个证候必须从患者的症状表现、脉、舌三方面考量，不能片面判断；
2.根据[证候推断]筛选出最有可能正确的一个或两个[核心证候]；
3.根据[核心证候]在[证候选项]中选出对应的[证候答案]；

注意：回答必须对应到[证候选项]中的选项对应的字母，筛选出的正确选项最多选2个。将答案输出在最后一行的“[证候答案]:”之后。


[历史病例]:
---------------------
{context_str}
---------------------


[核心临床信息]: {core_clinical_info}
[核心病机]: {mechanism_str}
[证候选项]: {syndrome_options}
[核心证候]: 
[证候答案]: 
"""
            qa_syndrome_infer_prompt_tmpl_str_temp = qa_syndrome_infer_prompt_tmpl_str.format(
                context_str="{context_str}",
                core_clinical_info=core_clinical_info_str,
                syndrome_options=syndrome_options,
                mechanism_str=mechanism_str)
            prompt3 = PromptTemplate(qa_syndrome_infer_prompt_tmpl_str_temp)
            rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index3, top_k,
                                                      hybrid_search, nodes3, with_hyde, qa_prompt_tmpl=prompt3)
            response = rag_query_engine.query(mechanism_str + core_clinical_info_str)
            for node in response.source_nodes:
                tools.printf(node.text)
            syndrome_answer = str(response)
            tools.printf(f"syndrome_answer:{syndrome_answer}")
            syndrome_answer = tools.select_answers_parse(syndrome_answer, "证候答案")
            tools.printf(syndrome_answer)
            syndrome_answer_str = tools.extract_core_mechanism(syndrome_options, syndrome_answer)
            tools.printf(syndrome_answer_str)
            # syndrome_answer = task3_ans[i].split(";")
            # syndrome_answer_str = tools.extract_core_mechanism(syndrome_options, syndrome_answer)
            # tools.printf(syndrome_answer_str)

        # task4
        qa_clinical_exp_prompt_tmpl_str_temp = qa_clinical_exp_prompt_tmpl_str.format(context_str="{context_str}",
                                                                                      clinical_info=clinical_info,
                                                                                      mechanism_str=mechanism_str)
        prompt4 = PromptTemplate(qa_clinical_exp_prompt_tmpl_str_temp)
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index4, top_k,
                                                  hybrid_search, nodes4, with_hyde, qa_prompt_tmpl=prompt4)
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
        # task4 = task4_ans[i]
        # clinical_experience_str = task4.split("辨证")[0].replace("临证体会：", "")
        # diagnosis_str = task4.split("辨证")[1].replace("辨证：", "").replace("：", "")

        resultAll = tools.printInfo(case_id, core_clinical_info_str, mechanism_answer, syndrome_answer,
                                    clinical_experience_str, diagnosis_str)
        tools.save_now(resultAll, submitPath)
        subResult.append(resultAll)
    end_time = time.time()
    print(f"start_time-end_time:{(end_time - start_time) / 60.0}min")
    tools.saveTxt("submit/subTemp.txt", subResult)

    ensemble.enseTask1(submitPath=submitPath, llmsubmit_path=submitPath, A_path=A_file)
