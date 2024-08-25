import re
import json


# 文本分割+llm抽取的答案（+llm筛除无用信息）
def enseTask1(submitPath, llmsubmit_path, A_path):
    # submitPath = "submit/sub21"  # 最终保存地址
    # llmsubmit_path = "submit/sub09.txt"  # 大模型生成答案的地址
    # A_path = "data/A榜.json"  # 临床信息所在位置
    with open(A_path, 'r', encoding='utf-8', errors='ignore') as file:
        clinical_info_data = json.load(file)
    with open(llmsubmit_path, 'r', encoding='utf-8', errors='ignore') as file:
        llmdata = file.readlines()
    resultList = []
    for i, t in enumerate(clinical_info_data):
        if i >= 50:
            break
        llmvalue = llmdata[i].split('@')[1].replace('\n', "").split(';')
        tempstr = t['临床资料'].replace(' ', "").replace('\n', "").replace("主诉及病史", "").replace("诊查", "")
        value = re.split("。|，|：|、|:|；", tempstr)  # 答案列表
        for l in llmvalue:
            if l in value:
                continue
            else:
                value.append(l)
        value_str = ""
        for v in value:
            if v == "":
                continue
            value_str += v + ';'
        reslist = llmdata[i].split('@')
        reslist[1] = value_str
        resultList.append(f"{reslist[0]}@{reslist[1]}@{reslist[2]}@{reslist[3]}@{reslist[4]}")
    with open(submitPath, "w", encoding="utf-8") as file:
        for item in resultList:
            file.write(item)


enseTask1(submitPath="submit/sub40.txt", llmsubmit_path="submit/sub39.txt", A_path="data/A榜.json")

