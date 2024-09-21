import json


def read_train_Json(input_file, k):
    # 打开并读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 遍历数据中的每个案例
    for i, case in enumerate(data):
        if i > k:
            break
        case_id = case["案例编号"]
        clinical_info = case["临床资料"]
        core_clinical_info = case["信息抽取能力-核心临床信息"]
        mechanism_inference = case["推理能力-病机推断"]
        core_mechanism = case["信息抽取能力-核心病机"]
        mechanism_answer = case["病机答案"]
        mechanism_options = case["病机选项"]
        syndrome_inference = case["推理能力-证候推断"]
        core_syndrome = case["信息抽取能力-核心证候"]
        syndrome_answer = case["证候答案"]
        syndrome_options = case["证候选项"]
        clinical_experience = case["临证体会"]
        diagnosis = case["辨证"]

        # 打印案例信息
        print(f"案例编号: {case_id}")
        print(f"临床资料: {clinical_info}")
        print(f"核心临床信息: {core_clinical_info}")
        print(f"病机推断: {mechanism_inference}")
        print(f"核心病机: {core_mechanism}")
        print(f"病机答案: {mechanism_answer}")
        print(f"病机选项: {mechanism_options}")
        print(f"证候推断: {syndrome_inference}")
        print(f"核心证候: {core_syndrome}")
        print(f"证候答案: {syndrome_answer}")
        print(f"证候选项: {syndrome_options}")
        print(f"临证体会: {clinical_experience}")
        print(f"辨证: {diagnosis}")
        print("-" * 80)


def read_AJson(input_file):
    # 打开并读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    infoList = []
    # 遍历数据中的每个案例
    for i, case in enumerate(data):
        case_id = case["案例编号"]
        clinical_info = case["临床资料"]
        mechanism_options = case["病机选项"]
        syndrome_options = case["证候选项"]
        infoList.append({"案例编号": case_id, "临床资料": clinical_info, "病机选项": mechanism_options,
                         "证候选项": syndrome_options})
    return infoList


# def split_json_file(input_file: str, split_nums: int = 4):
#     # 打开并读取JSON文件
#     with open(input_file, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     infoList = []
#     # 遍历数据中的每个案例
#     for i, case in enumerate(data):
#         case_id = case["案例编号"]
#         clinical_info = case["临床资料"]
#         mechanism_options = case["病机选项"]
#         syndrome_options = case["证候选项"]
#         infoList.append({"案例编号": case_id, "临床资料": clinical_info, "病机选项": mechanism_options,
#                          "证候选项": syndrome_options})
#     n = len(infoList)
#     step = int(infoList/split_nums)
#     b = [a[i:i+step] for i in range(0, l, step)]
#     for i in range(n):
#         with open(f"data/B_data_{i}.json", 'w', encoding='utf-8') as file:
#             json.dump(infoList, file, ensure_ascii=False, indent=4)
#     return infoList


def read_traintask1(input_file):
    # 打开并读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    infoList = []
    # 遍历数据中的每个案例
    for i, case in enumerate(data):
        clinical_info = case["[临床资料]"]
        core_clinical_info = case["[核心临床信息]"]
        infoList.append({"临床资料": clinical_info, "核心临床信息": core_clinical_info})
    return infoList


def read_traintask2(input_file):
    # 打开并读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    infoList = []
    # 遍历数据中的每个案例
    for i, case in enumerate(data):
        clinical_info = case["[临床资料]"]
        core_clinical_info = case["[核心临床信息]"]
        mechanism_options = case["[病机选项]"]
        mechanism_inference = case["[病机推断]"]
        core_mechanism = case["[核心病机]"]
        mechanism_answer = case["[病机答案]"]
        infoList.append(
            {"[临床资料]": clinical_info, "[核心临床信息]": core_clinical_info, "[病机选项]": mechanism_options,
             "[病机推断]": mechanism_inference, "[核心病机]": core_mechanism, "[病机答案]": mechanism_answer})
    return infoList


def read_traintask3(input_file):
    # 打开并读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    infoList = []
    # 遍历数据中的每个案例
    for i, case in enumerate(data):
        core_clinical_info = case["[核心临床信息]"]
        core_mechanism = case["[核心病机]"]
        syndrome_inference = case["[证候推断]"]
        core_syndrome = case["[核心证候]"]
        syndrome_answer = case["[证候答案]"]
        syndrome_options = case["[证候选项]"]
        infoList.append(
            {"[核心临床信息]": core_clinical_info, "[核心病机]": core_mechanism, "[证候选项]": syndrome_options,
             "[证候推断]": syndrome_inference, "[核心证候]": core_syndrome, "[证候答案]": syndrome_answer})
    return infoList


def read_traintask4(input_file):
    # 打开并读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    infoList = []
    # 遍历数据中的每个案例
    for i, case in enumerate(data):
        clinical_info = case["[临床资料]"]
        core_mechanism = case["[核心病机]"]
        clinical_experience = case["[临证体会]"]
        infoList.append({"[临床资料]": clinical_info, "[核心病机]": core_mechanism, "[临证体会]": clinical_experience})
    return infoList
