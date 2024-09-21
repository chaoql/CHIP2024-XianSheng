import re
import json
import os


def select_answers_parse(str, fstr):
    """
    截取字符串str中特定字符串fstr之后的答案，并删除首尾换行符和之间的空格
    """
    strtemp = str
    num = strtemp.rfind(fstr)
    pattern = r"[A-Z]"
    # 有的选项最后一个没有分号，匹配不上
    if num != -1:
        matches = re.findall(pattern, str[num + len(fstr):])
    else:
        matches = re.findall(pattern, str)
    if not matches:
        num = strtemp[0:num].rfind(fstr)
        matches = re.findall(pattern, str[num + len(fstr):])
    result_list = []
    for match in matches:
        if match.strip() not in result_list:
            result_list.append(match.strip())
    return result_list


def printf(obj):
    print(obj)
    print("-" * 40)


def printInfo_test(case_id, core_clinical_info, mechanism_answer, syndrome_answer, clinical_experience_str,
                   diagnosis_str, sorted_repeat_mechanism_dict, sorted_repeat_syndrome_dict):
    """
    打印当前对象的所有子任务结果
    :return:
    """
    mechanism_answer_str = ""
    for ma in mechanism_answer:
        mechanism_answer_str += ma + ';'
    mechanism_answer_str = mechanism_answer_str[:-1]
    syndrome_answer_str = ""
    for sa in syndrome_answer:
        syndrome_answer_str += sa + ';'
    syndrome_answer_str = syndrome_answer_str[:-1]
    print(
        f"\n{case_id}@{core_clinical_info}@{mechanism_answer_str}@{syndrome_answer_str}@临证体会：{clinical_experience_str}辨证：{diagnosis_str}@{sorted_repeat_mechanism_dict}@{sorted_repeat_syndrome_dict}\n")
    return f"{case_id}@{core_clinical_info}@{mechanism_answer_str}@{syndrome_answer_str}@临证体会：{clinical_experience_str}辨证：{diagnosis_str}@{sorted_repeat_mechanism_dict}@{sorted_repeat_syndrome_dict}"


def printInfo(case_id, core_clinical_info, mechanism_answer, syndrome_answer, clinical_experience_str, diagnosis_str):
    """
    打印当前对象的所有子任务结果
    :return:
    """
    mechanism_answer_str = ""
    for ma in mechanism_answer:
        mechanism_answer_str += ma + ';'
    mechanism_answer_str = mechanism_answer_str[:-1]
    syndrome_answer_str = ""
    for sa in syndrome_answer:
        syndrome_answer_str += sa + ';'
    syndrome_answer_str = syndrome_answer_str[:-1]
    print(
        f"\n{case_id}@{core_clinical_info}@{mechanism_answer_str}@{syndrome_answer_str}@临证体会：{clinical_experience_str}辨证：{diagnosis_str}\n")
    return f"{case_id}@{core_clinical_info}@{mechanism_answer_str}@{syndrome_answer_str}@临证体会：{clinical_experience_str}辨证：{diagnosis_str}"


def extract_core_mechanism(mechanism_options, mechanism_answer):
    """
    将病机选择对应到具体的病机项
    """
    mechanism_str = ""
    pattern = r"[A-Z]:(.*?);"
    # 有的选项最后一个没有分号，匹配不上
    matches = re.findall(pattern, mechanism_options + ';')

    # 将匹配结果转换为列表
    result_list = [match.strip() for match in matches]
    answer_list = mechanism_answer
    reAnswer_list = [m[0] for m in answer_list]
    for a in reAnswer_list:
        if ord(a) - ord('A') >= len(result_list):
            continue
        mechanism_str += (result_list[ord(a) - ord('A')] + ";")
    return mechanism_str


def saveTxt(submitPath, subResult):
    """
    保存提交结果。
    :return:
    """
    with open(submitPath, "w", encoding="utf-8") as file:
        # 遍历列表，将每个元素写入文件，每个元素占一行
        for item in subResult:
            file.write(item + "\n")


def save_now(result, path):
    with open(path, 'a', encoding="utf-8", errors="ignore") as file:
        file.write(result)
        file.write("\n")


def transfer_train_task1():
    # 打开并读取JSON文件
    with open("data/train.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
    infoList = []
    # 遍历数据中的每个案例
    for i, case in enumerate(data):
        clinical_info = case["临床资料"]
        core_clinical_info = case["信息抽取能力-核心临床信息"]
        infoList.append({"[临床资料]": clinical_info, "[核心临床信息]": core_clinical_info})
    with open("data/trainTask1.json", 'w', encoding='utf-8') as file:
        json.dump(infoList, file, ensure_ascii=False, indent=4)


def transfer_train_task2():
    # 打开并读取JSON文件
    with open("data/train.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
    infoList = []
    # 遍历数据中的每个案例
    for i, case in enumerate(data):
        clinical_info = case["临床资料"]
        core_clinical_info = case["信息抽取能力-核心临床信息"]
        mechanism_inference = case["推理能力-病机推断"].replace(":", "-推理病机为-")
        core_mechanism = case["信息抽取能力-核心病机"]
        infoList.append(
            {"[临床资料]": clinical_info, "[核心临床信息]": core_clinical_info, "[病机推断]": mechanism_inference,
             "[核心病机]": core_mechanism})
    with open("data/trainTask2WOoptions.json", 'w', encoding='utf-8') as file:
        json.dump(infoList, file, ensure_ascii=False, indent=4)


def transfer_train_task3():
    # 打开并读取JSON文件
    with open("data/train.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
    infoList = []
    # 遍历数据中的每个案例
    for i, case in enumerate(data):
        core_clinical_info = case["信息抽取能力-核心临床信息"]
        core_mechanism = case["信息抽取能力-核心病机"]
        syndrome_inference = case["推理能力-证候推断"].replace(":", "-推理证候为-")
        core_syndrome = case["信息抽取能力-核心证候"]
        infoList.append(
            {"[核心临床信息]": core_clinical_info, "[核心病机]": core_mechanism, "[证候推断]": syndrome_inference,
             "[核心证候]": core_syndrome, })
    with open("data/trainTask3WOoptions.json", 'w', encoding='utf-8') as file:
        json.dump(infoList, file, ensure_ascii=False, indent=4)


def transfer_train_task4():
    # 打开并读取JSON文件
    with open("data/train.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
    infoList = []
    # 遍历数据中的每个案例
    for i, case in enumerate(data):
        clinical_info = case["临床资料"]
        core_mechanism = case["信息抽取能力-核心病机"]
        clinical_experience = case["临证体会"][5:]
        infoList.append({"[临床资料]": clinical_info, "[核心病机]": core_mechanism, "[临证体会]": clinical_experience})
    with open("data/trainTask4.json", 'w', encoding='utf-8') as file:
        json.dump(infoList, file, ensure_ascii=False, indent=4)
