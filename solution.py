from custom.prompt import qa_clinical_info_prompt_tmpl_str, qa_core_mechanism_prompt_tmpl_str, \
    qa_syndrome_infer_prompt_tmpl_str, qa_clinical_exp_prompt_tmpl_str, choice_select_prompt_str, \
    refine_syndrome_tmpl_str
from llama_index.core import PromptTemplate
from pipeline import rag
import tools
from tqdm import *
from tools import saveTxt, printf
import os
from llama_index.core.postprocessor import LLMRerank
from llama_index.core import Settings
import json


def task1Solution(infoList, with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k, hybrid_search, nodes,
                  with_hyde, submitPath):
    resultList = []
    resnum = 0
    with open("data/train.json", 'r', encoding='utf-8', errors='ignore') as file:
        train_data = json.load(file)
    core_clinical_infos = []
    for i, data in enumerate(train_data):
        if i <= 50:
            continue
        core_info = data["信息抽取能力-核心临床信息"]
        core_info_list = core_info.split(";")
        for info in core_info_list:
            if info not in core_clinical_infos:
                core_clinical_infos.append(info)
    info_str = ""
    for info in core_clinical_infos:
        info_str += info + ";"
    if os.path.exists(submitPath + "-1.txt"):
        with open(submitPath + "-1.txt", 'r', encoding='utf-8', errors="ignore") as file:
            lines = file.readlines()
            resnum = len(lines)
    for i, info in tqdm(enumerate(infoList), total=len(infoList)):
        if i < resnum:
            continue
        case_id = info["案例编号"]
        clinical_info = info["临床资料"]
        new_clinical_info_prompt_tmpl_str = """
--------------
{context_str}
--------------
你是一名中医专家，请按照上述案例的格式回答当前临床资料中包含的[核心临床信息]，应包括病人的症状、舌、脉等信息。要求[核心临床信息]只列出以”;“分隔的核心信息，不含任何换行符等无关字符。
[临床资料]: {clinical_info}
[核心临床信息]: 
"""
        qa_clinical_info_prompt_tmpl_str_temp = new_clinical_info_prompt_tmpl_str.format(context_str="{context_str}",
                                                                                         clinical_info=clinical_info)
        prompt1 = PromptTemplate(qa_clinical_info_prompt_tmpl_str_temp)
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k,
                                                  hybrid_search, nodes, with_hyde, qa_prompt_tmpl=prompt1)
        response = rag_query_engine.query(clinical_info)
        core_clinical_info = str(response)
        printf(f"clinical_info:{clinical_info}")
        printf(f"core_clinical_info:{core_clinical_info}")

        all_clinical_info_prompt_tmpl_str = """请抽取如下临床资料中与临床相关的所有实体。答案只列出以”;“分隔的实体，不含任何换行符等无关字符。
[临床资料]: {clinical_info}
"""
        all_clinical_info_prompt_tmpl_str_temp = all_clinical_info_prompt_tmpl_str.format(clinical_info=clinical_info)
        prompt1_1 = PromptTemplate(all_clinical_info_prompt_tmpl_str_temp)
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k,
                                                  hybrid_search, nodes, with_hyde, qa_prompt_tmpl=prompt1_1)
        response1_1 = rag_query_engine.query(clinical_info)

        # from langchain.prompts import ChatPromptTemplate
        # prompt1 = ChatPromptTemplate.from_template(new_clinical_info_prompt_tmpl_str).format(context_str=info_str,
        #                                                                                      clinical_info=clinical_info)
        # response = Settings.llm.complete(prompt1)
        core_clinical_info1_1 = str(response1_1)
        printf(f"core_clinical_info1_1:{core_clinical_info1_1}")
        core_clinical_info_list = core_clinical_info.split(";")
        core_clinical_info1_1_list = core_clinical_info1_1.split(";")
        core_clinical_info_result = []
        core_clinical_info_str = ""
        for info in core_clinical_info_list:
            core_clinical_info1_1_list.append(info)
        for info in core_clinical_info1_1_list:
            if info not in core_clinical_info_result:
                core_clinical_info_result.append(info)
                core_clinical_info_str += info + ";"
        printf(f"core_clinical_info_str:{core_clinical_info_str}")
        resultAll = f"{case_id}@{core_clinical_info_str}"
        resultList.append(resultAll)
        tools.save_now(resultAll, submitPath + "-1.txt")
    # if resnum != len(infoList):
    #     with open(submitPath + "-1.txt", "w", encoding="utf-8") as file:
    #         for item in resultList:
    #             file.write(item + "\n")


def task2Solution(infoList, with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k, hybrid_search, nodes,
                  with_hyde, submitPath):
    resnum = 0
    lines = []
    if os.path.exists(submitPath + "-1.txt"):
        with open(submitPath + "-1.txt", 'r', encoding='utf-8', errors="ignore") as file:
            lines = file.readlines()
    if os.path.exists(submitPath + "-2.txt"):
        with open(submitPath + "-2.txt", 'r', encoding='utf-8', errors="ignore") as file:
            lines2 = file.readlines()
    else:
        lines2 = []
    if os.path.exists(submitPath + "-3.txt"):
        with open(submitPath + "-3.txt", 'r', encoding='utf-8', errors="ignore") as file:
            lines3 = file.readlines()
    resultList = []
    for i, info in tqdm(enumerate(infoList), total=len(infoList)):
        if i > 50:
            break
        case_id = info["案例编号"]
        clinical_info = info["临床资料"]
        mechanism_options = info["病机选项"]
        syndrome_options = info["证候选项"]
        core_clinical_info = lines[i].split("@")[1].rstrip("\n")
        syndrome_answer = lines3[i].split("@")[3].split(";")
        syndrome_answer_str = tools.extract_core_mechanism(syndrome_options, syndrome_answer)

        if lines2 and i < len(lines2):
            resnum += 1
            continue

        new_qa_core_mechanism_prompt_tmpl_str = """
        你是一名中医专家，请使用检索到的上下文信息，来根据患者的核心临床信息，完成病机推理。
病机的含义是指疾病的病因、病性、病位及病程中变化的要理。病机的内涵，在宏观整体层面上，大致可概括为邪正盛衰、阴阳失调、气血津液输化代谢失常。具体而言，又均由脏腑病变导致某个系统、某种疾病、某一证候及某个特异性症状、体征的病理表现，因而其类别有脏腑病机、疾病病机、证候病机、症状病机等多个方面，相互之间有其关联性、层次性，而最终必须落实在证候病机上，证候病机要素提供辨证信息，其又由内涵清楚、外延明确的病机证候要素条目构成。

病机推理的基本步骤：
1. 辨识病机证素。辨识病机证素是根据特异症、可见症和相关舌脉，识别病理因素及病位、病性。特异症是指人体内在病理变化表现在外的特征性症状、体征，是辨识病机证素的主要依据，即《伤寒论》所云“但见一症便是，不必悉具”之意。可见症是指人体内在病理变化可能表现的症状、体征，可因病而异。相关舌脉是辨识病机证素与脉症是否对应的参考依据。
以痹证为例，如风的特异症为关节疼痛游走不定、关节怕风；寒的特异症为关节冷痛、遇寒痛增、得热痛减、关节怕冷；湿的特异症为关节疼痛着而不移、关节痛阴雨天加重、肢体酸楚沉重。风的可见症为肢体肌肉疼痛酸楚、恶风、发热；寒的可见症为四肢清冷、关节拘痛；湿的可见症为关节漫肿、食欲不振、大便溏。风的舌脉表现为苔薄白，脉浮；寒表现为舌质淡或淡红，舌苔薄白，脉紧或迟；湿表现为舌苔腻，脉濡缓或细缓。临床上既可表现与病机一致的脉象，也可表现与病机不相一致的脉象，故又需根据具体情况舍脉从症或舍症从脉。
在明晰病理因素的基础上，确定病位、病性。痹证的病变在肢体关节，故脏腑病位主要在肝、肾、脾(肝主筋，肾主骨，脾主四肢肌肉；关节为骨之交接处，由筋膜束合而成)。若病初以关节、肌肉疼痛为主，则病在肌表经络；病久以关节变形、僵痛为主则深入筋骨，病及肝肾；兼有肌肉瘦削，则病及于脾。依据中医基础理论，综合特异症、可见症和相关舌脉即可判断病性的阴阳虚实、标本缓急。
2. 根据病机证素的组合确定证名。病机证素是辨证诊断的基本单元，多为脏腑病机、病理因素之间的兼夹、复合，如肾虚肝郁、肝郁脾虚、瘀热相搏、湿热郁蒸、寒湿痹阻、痰热内蕴、痰湿中阻、风火相扇等皆为临床常见的兼夹、复合病机，交叉组合而成的证候名称。
疾病总是处于不断的变化之中，临证必须注意病机的动态演变，围绕病机之间的兼夹、复合和转化、演变规律进行分析、归纳，根据各种疾病的不同，明晰病机证素的分类、组合特点，能直接指导临床的辨证论治，提高临床疗效。如“瘀热”病机学说为我们长期从事临床科学研究的理论创新，认为“瘀热”是多种外感、内伤疾病的病变过程中所产生的一种复合病理因素，由血热、血瘀两种病理因素互为搏结、相合为患而形成，临床表现为“瘀热相搏证”，由于疾病的不同又可表现为不同的子证，如中风的瘀热阻窍证、重症肝炎的瘀热发黄证、急性肾衰的瘀热水结证、各种出血性疾病的瘀热血溢证。再如痹证初起多表现为风湿、风寒湿、风湿热痹阻(风寒湿痹证、风湿热痹证等)；三者之间又可转化、兼夹， 表现为风寒湿热痹 (寒热错杂证)；病久还可表现为痰瘀痹阻、肝肾气血亏虚(痰瘀互结证、肝肾亏虚证、气血亏虚证)。

对于当前问题，你的推理过程应该是：
1. 使用检索到的上下文，根据当前患者的[核心临床信息]推理当前患者的核心病机；
2. 统计[核心病机]的数量，答案的数量不超过3个；
3. 根据[核心病机]在[病机选项]中选出对应的[病机答案]，不能漏选或多选；如果找不到病机对应的选项，就选择含义最相近的；
4. 将最终正确的选项输出在最后一行“[病机答案]:”之后。

--上下文--
{context_str}

--当前患者--
[核心临床信息]: {core_clinical_info}
[病机选项]: {mechanism_options}
[病机推断]: 
[核心病机]: 
[病机答案]: 
"""
        qa_core_mechanism_prompt_tmpl_str = """你是一名中医专家，请使用上下文信息，根据抽取到的核心临床信息，完成病机推理。

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
        qa_core_mechanism_prompt_tmpl_str_temp = qa_core_mechanism_prompt_tmpl_str.format(
            context_str="{context_str}",
            core_clinical_info=core_clinical_info,
            mechanism_options=mechanism_options,
            syndrome_answer_str=syndrome_answer_str,
        )
        prompt2 = PromptTemplate(qa_core_mechanism_prompt_tmpl_str_temp)
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k,
                                                  hybrid_search, nodes, with_hyde, qa_prompt_tmpl=prompt2)
        response = rag_query_engine.query(syndrome_answer_str)
        # for node in response.source_nodes:
        #     tools.printf(node.text)
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
        tools.save_now(f"{case_id}@{core_clinical_info}@{mechanism_answer_str}",
                       submitPath + "-2.txt")
    if resnum != len(infoList):
        with open(submitPath + "-2_temp.txt", "w", encoding="utf-8") as file:
            for item in resultList:
                file.write(item + '\n')


def task3Solution(infoList, with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k, hybrid_search, nodes,
                  with_hyde, submitPath):
    resnum = 0
    if os.path.exists(submitPath + "-2.txt"):
        with open(submitPath + "-2.txt", 'r', encoding='utf-8', errors="ignore") as file:
            lines = file.readlines()
            resnum = len(lines)
    else:
        lines = []
    if os.path.exists(submitPath + "-3.txt"):
        with open(submitPath + "-3.txt", 'r', encoding='utf-8', errors="ignore") as file:
            lines2 = file.readlines()
    else:
        lines2 = []
    resultList = []
    for i, info in tqdm(enumerate(infoList), total=len(infoList)):
        if i > 50:
            break
        if i < len(lines2):
            continue
        case_id = info["案例编号"]
        mechanism_options = info["病机选项"]
        syndrome_options = info["证候选项"]
        core_clinical_info = lines[i].split("@")[1]
        mechanism_answer_str = lines[i].split("@")[2].strip()
        mechanism_answer = mechanism_answer_str.split(";")

        mechanism_str = tools.extract_core_mechanism(mechanism_options, mechanism_answer)
        #         qa_syndrome_infer_prompt_tmpl_str = """
        # 你是一名中医专家，请使用中医辨证知识，根据患者[核心临床信息]完成[证候推断]。
        #
        # 这是一道选择题，你的推理过程应该是：
        # 1.根据[中医辨证知识]和[核心临床信息]进行[证候推断]，每个证候必须从患者的症状表现、脉、舌三方面考量，不能片面判断；
        # 2.根据[证候推断]筛选出最有可能正确的一个或两个[核心证候]；
        # 3.根据[核心证候]在[证候选项]中选出对应的[证候答案]；
        #
        # 注意：回答必须对应到[证候选项]中的选项对应的字母，筛选出的正确选项最多选2个。将答案输出在最后一行的“[证候答案]:”之后。
        #
        #
        # [中医辨证知识]:
        # ---------------------
        # {context_str}
        # ---------------------
        #
        #
        # [核心临床信息]: {core_clinical_info}
        # [证候选项]: {syndrome_options}
        # [核心证候]:
        # [证候答案]:
        # """
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
        qa_syndrome_infer_prompt_tmpl_str_temp = qa_syndrome_infer_prompt_tmpl_str.format(context_str="{context_str}",
                                                                                          core_clinical_info=core_clinical_info,
                                                                                          syndrome_options=syndrome_options,
                                                                                          mechanism_str=mechanism_str,
                                                                                          )
        prompt3 = PromptTemplate(qa_syndrome_infer_prompt_tmpl_str_temp)

        # simple
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k,
                                                  hybrid_search, nodes, with_hyde, qa_prompt_tmpl=prompt3)
        response = rag_query_engine.query(mechanism_str + core_clinical_info)
        for node in response.source_nodes:
            tools.printf(node.text)
        syndrome_answer = str(response)
        tools.printf(f"syndrome_answer:{syndrome_answer}")
        syndrome_answer = tools.select_answers_parse(syndrome_answer, "[证候答案]")
        tools.printf(syndrome_answer)
        syndrome_answer_str = tools.extract_core_mechanism(syndrome_options, syndrome_answer)
        tools.printf(syndrome_answer_str)

        syndrome_answer_str = ""
        for sa in syndrome_answer:
            syndrome_answer_str += sa + ';'
        syndrome_answer_str = syndrome_answer_str[:-1]
        resultList.append(f"{case_id}@{core_clinical_info}@{mechanism_answer_str}@{syndrome_answer_str}")
        tools.save_now(f"{case_id}@{core_clinical_info}@{mechanism_answer_str}@{syndrome_answer_str}",
                       submitPath + "-3.txt")
    with open(submitPath + "-3_temp.txt", "w", encoding="utf-8") as file:
        for item in resultList:
            file.write(item + '\n')


def task4Solution(infoList, with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k, hybrid_search, nodes,
                  with_hyde, submitPath):
    resnum = 0
    lines = []
    if os.path.exists(submitPath + "-3.txt"):
        with open(submitPath + "-3.txt", 'r', encoding='utf-8', errors="ignore") as file:
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
        syndrome_answer_str = linsI[3].rstrip()
        syndrome_answer = syndrome_answer_str.split(";")
        syndrome_str = tools.extract_core_mechanism(syndrome_options, syndrome_answer)
        qa_clinical_exp_prompt_tmpl_str = """
你是一名中医专家，请参考检索到的上下文和案例，根据患者的[临床资料]和[核心病机]撰写辨证摘要，即[临证体会]。
这是一个辨别证型的一个思考过程，突出分析病机的过程。
要求语言简洁凝练，尽量用一两句话总结。

---------------------
{context_str}
---------------------

[核心临床信息]: {core_clinical_info}
[核心病机]: {mechanism_str}
[临证体会]: 
"""
        qa_clinical_exp_prompt_tmpl_str_temp = qa_clinical_exp_prompt_tmpl_str.format(context_str="{context_str}",
                                                                                      core_clinical_info=core_clinical_info,
                                                                                      mechanism_str=mechanism_str)
        prompt4 = PromptTemplate(qa_clinical_exp_prompt_tmpl_str_temp)
        rag_query_engine = rag.Build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode, index, top_k,
                                                  hybrid_search, nodes, with_hyde, qa_prompt_tmpl=prompt4)
        response = rag_query_engine.query(mechanism_str + syndrome_str)
        clinical_experience_str = str(response).replace("\n", "")
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
        tools.save_now(f"{case_id}@{core_clinical_info}@{mechanism_answer_str}@{syndrome_answer_str}@临证体会：{clinical_experience_str}辨证：{diagnosis_str}",
                       submitPath + "-4.txt")
    with open(submitPath + "-4.txt", "w", encoding="utf-8") as file:
        for item in resultList:
            file.write(item + '\n')
