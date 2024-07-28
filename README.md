# CHIP2024
CHIP2024-中医辨证思维测评

trainTask1~4为每个任务的专项检索增强生成数据库（其中trainTask2/3WOoptions为去除选项直接检索显示答案的任务2/3专项检索增强生成数据库）。

trainTaskTXT1~4为每个任务的专项检索增强生成数据库(文本类型版本)。

store01~04为按照json格式解析的普通检索向量数据库，根据trainTask1、trainTask2WOoptions、trainTask3WOoptions、trainTask4生成。

store05~08为按照文本格式解析的混合检索向量数据库，根据trainTaskTXT1~4生成。