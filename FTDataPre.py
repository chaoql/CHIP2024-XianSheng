import json


def task1Prepare(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    infoList = []
    # 遍历数据中的每个案例
    for i, case in enumerate(data):
        clinical_info = case["临床资料"]
        core_clinical_info = case["信息抽取能力-核心临床信息"]
        systemInfo = {"role": "system", "content": "你是一位乐于助人，知识渊博的中医专家。"}
        roleQuery = {"role": "user", "content": f"请抽取临床资料中的核心临床信息。临床资料：{clinical_info}"}
        roleAnswer = {"role": "assistant", "content": f"{core_clinical_info}"}
        infoList.append({"messages": [systemInfo, roleQuery, roleAnswer]})
    # 将字典列表转换为JSON格式的字符串
    json_string = json.dumps(infoList, separators=(',', ':'), ensure_ascii=False)

    # 将JSON字符串按对象分割成列表，每个对象独占一行
    lines = json_string.split(']},{')

    with open(output_path, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line+"]}\n{")


task1Prepare("data/train.json", "FTdata/FTDataTask1.json")
