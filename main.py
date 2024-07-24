from dotenv import load_dotenv, find_dotenv
from dotenv import dotenv_values
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core import Settings
from custom.glmfz import ChatGLM
from langchain.prompts import ChatPromptTemplate
from custom.prompt import clinical_info_prompt_tmpl_str, core_mechanism_prompt_tmpl_str, syndrome_infer_prompt_tmpl_str, \
    clinical_exp_prompt_tmpl_str
import re
import pipeline.readData as readData
import tools

if __name__ == '__main__':
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
    response_mode = ResponseMode.COMPACT  # 最佳实践为为TREE_SUMMARIZE
    submitPath = "submit/sub01.txt"
    A_file = "data/A榜.json"
    infoList = readData.read_AJson(A_file)
    subResult = []
    for info in infoList:
        case_id = info["案例编号"]
        clinical_info = info["临床资料"]
        mechanism_options = info["病机选项"]
        syndrome_options = info["证候选项"]
        # task1
        prompt1 = ChatPromptTemplate.from_template(clinical_info_prompt_tmpl_str).format(clinical_info=clinical_info)
        result1 = Settings.llm.complete(prompt1)
        core_clinical_info = result1.text
        # printf(f"core_clinical_info:{core_clinical_info}")

        # task2
        prompt2 = ChatPromptTemplate.from_template(core_mechanism_prompt_tmpl_str).format(
            clinical_info=clinical_info,
            core_clinical_info=core_clinical_info,
            mechanism_options=mechanism_options
        )
        result2 = Settings.llm.complete(prompt2)
        # printf(result2.text)
        mechanism_answer = tools.select_answers_parse(result2.text, "[病机答案]:")
        # printf(mechanism_answer)
        mechanism_str = tools.extract_core_mechanism(mechanism_options, mechanism_answer)
        # printf(mechanism_str)

        # task3
        prompt3 = ChatPromptTemplate.from_template(syndrome_infer_prompt_tmpl_str).format(
            core_clinical_info=core_clinical_info,
            syndrome_options=syndrome_options,
            mechanism_str=mechanism_str
        )
        result3 = Settings.llm.complete(prompt3)
        # printf(result3.text)
        syndrome_answer = tools.select_answers_parse(result3.text, "[证候答案]:")
        # printf(syndrome_answer)
        syndrome_answer_str = tools.extract_core_mechanism(syndrome_options, syndrome_answer)
        # printf(syndrome_answer_str)

        # task4
        prompt4 = ChatPromptTemplate.from_template(clinical_exp_prompt_tmpl_str).format(
            clinical_info=clinical_info,
            mechanism_str=mechanism_str,
        )
        result4 = Settings.llm.complete(prompt4)
        # printf(result4.text)
        clinical_experience_str = result4.text

        diagnosis = syndrome_answer_str.split(";")
        diagnosis_str = ""
        for d in diagnosis:
            if d:
                diagnosis_str += d + '，'
        diagnosis_str = diagnosis_str[:-1]
        # printf(diagnosis_str)

        resultAll = tools.printInfo(case_id, core_clinical_info, mechanism_answer, syndrome_answer, clinical_experience_str,
                              diagnosis_str)
        subResult.append(resultAll)
    tools.saveTxt(submitPath, subResult)

    # index, nodes = rag.load_data(train_file, persist_dir)
    # rag_query_engine, simple_query_engine = rag.Build_query_engine(index, top_k, hybrid_search, nodes, with_hyde)
    # response = rag_query_engine.query(query_str)
