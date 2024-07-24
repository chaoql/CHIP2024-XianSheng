import re


def select_answers_parse(str, fstr):
    """
    截取字符串str中特定字符串fstr之后的答案，并删除首尾换行符和之间的空格
    """
    num = str.find(fstr)
    pattern = r"[A-Z]"
    # 有的选项最后一个没有分号，匹配不上
    matches = re.findall(pattern, str[num + len(fstr):])

    result_list = []
    for match in matches:
        if match.strip() not in result_list:
            result_list.append(match.strip())
    return result_list


def printf(obj):
    print(obj)
    print("-" * 40)


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
        f"{case_id}@{core_clinical_info}@{mechanism_answer_str}@{syndrome_answer_str}@临证体会：{clinical_experience_str}辨证：{diagnosis_str}")
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
