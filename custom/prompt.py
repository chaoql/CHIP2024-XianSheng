refine_mechanism_tmpl_str = """你是一个中医专家，当前有一个患者，其核心临床信息为: {query_str}
根据其核心临床信息，我们提供一个推理得到的现有的病机答案: {existing_answer}。
现在，你有机会去优化这个答案。仅在必要的情况下，下面给出一些中医案例和上下文信息。这些信息仅供参考，答案应该只根据最初核心临床信息进行判断。
------------
{context_msg}
------------
根据新的上下文优化当前答案，以更好地回答问题。但如果你认为给出的案例和上下文不足以优化当前答案，请返回现有的病机答案选项。
解决问题的步骤为：1. 根据当前患者的核心临床信息分析现有的病机答案；2. 根据历史案例和上下文信息优化答案，优化后的答案最多选3个；3. 输出格式化，把优化后的答案对应到选项，并放在 <优化后的答案> 之后。答案只包含正确选项。
当前病机选项: {options}
<优化后的答案>: 
"""
refine_syndrome_tmpl_str = """你是一个中医专家，当前有一个患者，其核心临床信息为: {query_str}。核心病机为{mechanism}
根据其核心临床信息和核心病机，我们提供一个推理得到的现有的证候答案: {existing_answer}。
现在，你有机会去优化这个答案。仅在必要的情况下，下面给出一些中医案例和上下文信息。这些信息仅供参考，答案应该只根据最初核心临床信息进行判断。
------------
{context_msg}
------------
根据新的上下文优化当前答案，以更好地回答问题。但如果你认为给出的案例和上下文不足以优化当前答案，请返回现有的病机答案选项。
解决问题的步骤为：1. 根据当前患者的核心临床信息和核心病机分析现有的证候答案；2. 根据历史案例和上下文信息优化答案，优化后的答案最多选2个；3. 输出格式化，把优化后的答案对应到选项，并放在 <优化后的答案> 之后。答案只包含正确选项。
当前证候选项: {options}
<优化后的答案>: 
"""
choice_select_prompt_str = """文件列表如下。每份文件旁边都有一个编号和文件摘要。还提供了一句描述。
请按相关性顺序回答问题时应参考的文件编号以及相关性得分。相关性评分是一个1到10分的数字，基于您认为文件与这句描述的相关程度。
不要包含任何与描述无关的文件。
格式示例：
文档1: 
<文档1的摘要>

文档2: 
<文档2的摘要>

...

文档10: 
<文档10的摘要>

描述: <描述>
答案: 
文件: 9, 相关性: 7
文件: 3, 相关性: 4
文件: 7, 相关性: 3

答案应与示例格式一致。现在让我们试试这个:

{context_str}
描述: {query_str}
答案: 
"""
new_clinical_info_prompt_tmpl_str = """
--------------
{context_str}
--------------
你是一名中医专家，现在对患者进行辨证诊断，请按照上述案例的格式回答当前[临床资料]中包含的[核心临床信息]，一般包括患者的症状、体征、病史等。要求[核心临床信息]只列出以”;“分隔的核心信息，不含任何换行符等无关字符。
注意：[核心临床信息]只包含辨证诊断过程中的重要信息，比如嗳气（临床信息）可以推理病机为胃气上逆（病机），并根据病机推断出证候为肝胃不和（证候）。次要信息及无关信息不在标注范围内，如“舌红苔白”，并非辨证诊断过程中的重要信息，不予标注。[核心临床信息]字段尽可能完整，比如“胸骨后及胃脘部胀痛”，仅标注“胀痛”会存在重要信息丢失。
临床信息与病机和病机与证候之间分别存在推断关系，比如从临床资料中提取临床信息“嗳气”、“抑郁”分别进行病机推断，推断出病机“胃气上逆”、“肝气郁结”，综合病机进行证候推断，推断出证候“肝胃不和”。
针对多个实体连在一起的长mention，本任务按照如下规则进行标注：如果每个实体具备独立意义则分开标注，如：“胸骨后及胃脘部胀痛，胸骨后有灼热感，吞咽时有梗噎感，伴嗳气、恶心”中标注（"胸骨后及胃脘部胀痛"，"胸骨后有灼热感"，"吞咽时有梗噎感"伴"嗳气"，"恶心"）。
[临床资料]: {clinical_info}
[核心临床信息]: 
"""

qa_clinical_info_prompt_tmpl_str = """
--------------
{context_str}
--------------
你是一名中医专家，请按照上述案例的格式回答当前临床资料中包含的[核心临床信息]。要求[核心临床信息]只列出以”;“分隔的核心信息，不含任何换行符等无关字符。
[临床资料]: {clinical_info}
[核心临床信息]: 
"""

clinical_info_prompt_tmpl_str = """
你是一名中医专家，请按照如下案例的方式进行抽取核心临床信息：
--------------
临床资料: 安某,女,50岁。初诊:1983年6月20日。主诉及病史:近40天来,胸骨后及胃脘部胀痛,胸骨后有灼热感,吞咽时有梗噎感,伴嗳气、恶心、泛酸,时呕吐出食物,纳食差,大便秘结。进寒冷食物时疼痛加剧,周身疲乏无力,经用中西药治疗无明显效果。诊查:现面色无华。舌质淡红、舌苔薄黄,脉弦滑略数,经钡餐透视检查诊为可复性食管裂孔疝,反流性食管炎。
核心临床信息: 胸骨后及胃脘部胀痛;胸骨后有灼热感;吞咽时有梗噎感;嗳气;恶心;泛酸;呕吐;纳食差;便秘结;舌苔薄黄;脉弦滑略数
--------------
临床资料: 王某，男，38岁。初诊:1979年9月12日。主诉及病史:鼻流血15天。8月17日，突然鼻流血，最严重的一天流血5次，每次约100~300ml，继而时出时止，头昏晕痛，口渴鼻干，胸闷气逆，大便干。诊查:脉浮大数、84次/分，舌苔薄白、舌质红。
核心临床信息: 鼻流血;口渴鼻干;胸闷气逆;大便干;脉浮大数
--------------
临床资料: 谭某，男，9岁:初诊:1977年1月8日。主诉及病史:患儿于1岁9个月时突然发热、浮肿，当时诊为急性肾炎。以后曾在多所医院住院治疗，诊为慢性肾炎，曾用中西药物而疗效不显。后来家长失去信心，不再予治疗。1976年12月底，患儿发热、咳嗽，以后出现嗜睡、鼻衄、恶心、呕吐、尿少，于1977年1月某日急诊入院。入院时体检:明显消瘦，皮肤干燥，鼻翼煽动，呼吸困难，心律不齐。实验室检查:二氧化碳结合力12.2容积%，尿素氮216mg%，血色素5.8g，诊断为慢性肾炎、尿毒症、酸中毒、继发性贫血。入院后立即输液、纠正酸中毒及脱水;予抗生素和中药真武汤、生脉散加味方，症状稍有稳定，二氧化碳结合力上升至56容积%，但全身症状无大改善，仍处于嗜睡衰竭状态，同时有鼻衄、呕吐咖啡样物。1月6日血色素降至4.5g，当时曾予输血。1月7日患儿情况转重，不能饮食，恶心呕吐频频发作，服药亦十分困难;大便1日数次，呈柏油样便;并有呕血、呼吸慢而不整(14~18次/分)，心率减至60~65次/分，当即予可拉明、洛贝林、生脉散注射液交替注射。1月8日，患儿继续呈嗜睡衰竭状态，面色晦暗，呼吸减慢，心率减慢至60次/分，大便仍为柏油便，情况越来越重，因急请会诊。诊查:会诊时，患儿呈嗜睡朦胧状态，时有恶心呕吐，呼吸深长而慢，脉沉细微弱无力而迟，舌嫩润齿痕尖微赤，苔薄白干中心微黄。
核心临床信息: 不能饮;呈柏油样便;呕血;嗜睡衰竭;面色晦暗;恶心呕吐;脉沉细微弱无力而迟;舌嫩润齿痕尖微赤;苔薄白干中心微黄
--------------
临床资料: {clinical_info}
核心临床信息: 
"""

qa_core_mechanism_prompt_tmpl_str = """
你是一名中医专家，请以如下案例的方式根据抽取到的核心临床信息，完成病机推理。
你的推理过程应该是：1.根据[核心临床信息]进行[病机推断]；2.根据[病机推断]筛选出[核心病机]；3.筛选出最有可能正确的数个[核心病机]；4.根据[核心病机]在[病机选项]中选出对应的[病机答案]。
注意：筛选出的正确选项最多选三个。要求：答案只包含正确选项，不含任何误导性。
---------------------
{context_str}
---------------------
[临床资料]: {clinical_info}
[核心临床信息]: {core_clinical_info}
[病机选项]: {mechanism_options}
[病机推断]: 
[核心病机]: 
[病机答案]: 
"""

core_mechanism_prompt_tmpl_str = """
你是一名中医专家，请以如下案例的方式根据抽取到的核心临床信息，完成病机推理。
你的推理过程应该是：1.根据[核心临床信息]进行[病机推断]；2.根据[病机推断]筛选出[核心病机]；3.根据[核心病机]在[病机选项]中选出对应的[病机答案]。
--------------
[临床资料]: 王某，男，38岁。初诊:1979年9月12日。主诉及病史:鼻流血15天。8月17日，突然鼻流血，最严重的一天流血5次，每次约100~300ml，继而时出时止，头昏晕痛，口渴鼻干，胸闷气逆，大便干。诊查:脉浮大数、84次/分，舌苔薄白、舌质红。
[核心临床信息]: 鼻流血;口渴鼻干;胸闷气逆;大便干;脉浮大数
[病机选项]: A:肝郁;B:伤阴耗气;C:湿邪阻滞;D:阴虚阳亢;E:耗伤心神;F:损伤中气;G:扰动神明;H:血热不固;I:水饮内停;J:热伤肺络
[病机推断]: 鼻流血-推理病机为-血热不固;口渴鼻干-推理病机为-热伤肺络;大便干-推理病机为-热伤肺络;胸闷气逆-推理病机为-热伤肺络;脉浮大数-推理病机为-热伤肺络
[核心病机]: 热伤肺络;血热不固
[病机答案]: H;J
--------------
[临床资料]: 谭某，男，9岁:初诊:1977年1月8日。主诉及病史:患儿于1岁9个月时突然发热、浮肿，当时诊为急性肾炎。以后曾在多所医院住院治疗，诊为慢性肾炎，曾用中西药物而疗效不显。后来家长失去信心，不再予治疗。1976年12月底，患儿发热、咳嗽，以后出现嗜睡、鼻衄、恶心、呕吐、尿少，于1977年1月某日急诊入院。入院时体检:明显消瘦，皮肤干燥，鼻翼煽动，呼吸困难，心律不齐。实验室检查:二氧化碳结合力12.2容积%，尿素氮216mg%，血色素5.8g，诊断为慢性肾炎、尿毒症、酸中毒、继发性贫血。入院后立即输液、纠正酸中毒及脱水;予抗生素和中药真武汤、生脉散加味方，症状稍有稳定，二氧化碳结合力上升至56容积%，但全身症状无大改善，仍处于嗜睡衰竭状态，同时有鼻衄、呕吐咖啡样物。1月6日血色素降至4.5g，当时曾予输血。1月7日患儿情况转重，不能饮食，恶心呕吐频频发作，服药亦十分困难;大便1日数次，呈柏油样便;并有呕血、呼吸慢而不整(14~18次/分)，心率减至60~65次/分，当即予可拉明、洛贝林、生脉散注射液交替注射。1月8日，患儿继续呈嗜睡衰竭状态，面色晦暗，呼吸减慢，心率减慢至60次/分，大便仍为柏油便，情况越来越重，因急请会诊。诊查:会诊时，患儿呈嗜睡朦胧状态，时有恶心呕吐，呼吸深长而慢，脉沉细微弱无力而迟，舌嫩润齿痕尖微赤，苔薄白干中心微黄。
[核心临床信息]: 不能饮;呈柏油样便;呕血;嗜睡衰竭;面色晦暗;恶心呕吐;脉沉细微弱无力而迟;舌嫩润齿痕尖微赤;苔薄白干中心微黄
[病机选项]: A:热结阳明;B:风阳上升;C:疫毒乘虚内侵中焦;D:冲任受损;E:气血两虚;F:心移热于小肠;G:气阴两虚;H:脾胃败绝;I:肠道传导失司;J:痰浊
[推理能力-病机推断]: 苔薄白干中心微黄-推理病机为-气阴两虚;舌嫩润齿痕尖微赤-推理病机为-气阴两虚;嗜睡衰竭-推理病机为-气阴两虚;呈柏油样便-推理病机为-脾胃败绝;呕血-推理病机为-脾胃败绝;面色晦暗-推理病机为-脾胃败绝;恶心呕吐-推理病机为-脾胃败绝;不能饮-推理病机为-脾胃败绝;脉沉细微弱无力而迟-推理病机为-气阴两虚
[核心病机]: 脾胃败绝;气阴两虚
[病机答案]: G;H    
--------------
[临床资料]: 张某，男，38岁。初诊：1985年4月18日。主诉及病史：右耳因爆竹震聋，时历2个月。自感鸣响不息，耳边如有高声，则耳内倍觉不舒。西医诊断为“爆炸性耳聋”，经治未见好转。诊查：鼓膜完整，标志存在。音叉测验正常。舌薄苔，脉有弦意。
[核心临床信息]: 右耳因爆竹震聋;耳边如有高声;耳内倍觉不舒;脉有弦意
[病机选项]: A:津不上承;B:卫阳不能外固;C:热邪下注;D:相火不藏;E:脾气不运;F:痰浊夹肝风上逆;G:感受署邪;H:耗损心气和心阴;I:肝肺被伤;J:气虚
[病机推断]: 右耳因爆竹震聋-推理病机为-肝肺被伤;耳边如有高声-推理病机为-肝肺被伤;耳内倍觉不舒-推理病机为-肝肺被伤;脉有弦意-推理病机为-肝肺被伤
[核心病机]: 肝肺被伤
[病机答案]: I
--------------
[临床资料]: {clinical_info}
[核心临床信息]: {core_clinical_info}
[病机选项]: {mechanism_options}
[病机推断]: 
[核心病机]: 
[病机答案]: 
"""

qa_syndrome_infer_prompt_tmpl_str = """
你是一名中医专家，请根据过往相关案例信息，在没有先验知识的情况下，根据[核心病机]，完成[证候推断]。
你的推理过程应该是：1.根据[核心病机]进行[证候推断]；2.根据[证候推断]筛选出最有可能正确的数个[核心证候]；3.根据[核心证候]在[证候选项]中选出对应的[证候答案]。
注意：筛选出的正确选项最多选两个。要求：答案只包含正确选项，不含任何误导性。
---------------------
{context_str}
---------------------
[核心临床信息]: {core_clinical_info}
[核心病机]: {mechanism_str}
[证候选项]: {syndrome_options}
[证候推断]: 
[核心证候]: 
[证候答案]: 
"""

syndrome_infer_prompt_tmpl_str = """
你是一名中医专家，请以如下案例的方式根据核心病机，完成证候推断。要求[证候答案]只列出以”;“分隔的正确选项，不含其他任何字符。
你的推理过程应该是：1.根据[核心病机]进行[证候推断]；2.根据[证候推断]筛选出[核心证候]；3.根据[核心证候]在[证候选项]中选出对应的[证候答案]。
--------------
[核心临床信息]: 鼻流血;口渴鼻干;胸闷气逆;大便干;脉浮大数
[核心病机]: 热伤肺络;血热不固
[证候选项]: A:脾胃不和;B:血热妄行;C:心肾两亏;D:湿热互结;E:邪陷心包;F:暑温动风;G:脾肾阳虚;H:痰蒙心窍;I:热伤阳络;J:痰湿内蕴
[证候推断]: 热伤肺络-推理证候为-热伤阳络;血热不固-推理证候为-血热妄行
[核心证候]: 热伤阳络;血热妄行
[证候答案]: B;I
--------------
[核心临床信息]: 不能饮;呈柏油样便;呕血;嗜睡衰竭;面色晦暗;恶心呕吐;脉沉细微弱无力而迟;舌嫩润齿痕尖微赤;苔薄白干中心微黄
[核心病机]: 脾胃败绝;气阴两虚
[证候选项]: A:肝肾同病;B:气阴两竭;C:气血亏虚;D:瘀血阻络;E:气血失调;F:郁滞胸脘;G:心阳不振;H:本虚标实;I:气血两亏;J:瘀血凝滞
[证候推断]: 脾胃败绝-推理证候为-气阴两竭;气阴两虚-推理证候为-气阴两竭
[核心证候]: 气阴两竭
[证候答案]: B
--------------
[核心临床信息]: 右耳因爆竹震聋;耳边如有高声;耳内倍觉不舒;脉有弦意
[核心病机]: 肝肺被伤
[证候选项]: A:脾运不畅;B:阴竭阳脱;C:脾肾虚衰;D:阴虚火旺;E:胃中虚冷;F:气滞血瘀;G:下虚失摄;H:胞脉瘀阻;I:心脾肾虚;J:热利伤津
[证候推断]: 肝肺被伤-推理证候为-气滞血瘀
[核心证候]: 气滞血瘀
[证候答案]: F
--------------
[核心临床信息]: {core_clinical_info}
[核心病机]: {mechanism_str}
[证候选项]: {syndrome_options}
[证候推断]: 
[核心证候]: 
[证候答案]: 
"""

qa_clinical_exp_prompt_tmpl_str = """
你是一名中医专家，请根据过往相关案例信息，在没有先验知识的情况下，根据[临床资料]和[核心病机]得出[临证体会]。要求[临证体会]只有一句话，且不能包含换行符。
---------------------
{context_str}
---------------------
[临床资料]: {clinical_info}
[核心病机]: {mechanism_str}
[临证体会]: 
"""

clinical_exp_prompt_tmpl_str = """
你是一名中医专家，请以如下案例的方式根据[临床资料]和[核心病机]得出[临证体会]。要求[临证体会]用一句话总结，不包含其他字符。
--------------
[临床资料]: 王某，男，38岁。初诊:1979年9月12日。主诉及病史:鼻流血15天。8月17日，突然鼻流血，最严重的一天流血5次，每次约100~300ml，继而时出时止，头昏晕痛，口渴鼻干，胸闷气逆，大便干。诊查:脉浮大数、84次/分，舌苔薄白、舌质红。
[核心病机]: 热伤肺络;血热不固
[临证体会]: 患者素体健康，今突发鼻衄，观其来势之迅猛，衄血量之多，实系热伤肺络、血热不固。
--------------
[临床资料]: 谭某，男，9岁:初诊:1977年1月8日。主诉及病史:患儿于1岁9个月时突然发热、浮肿，当时诊为急性肾炎。以后曾在多所医院住院治疗，诊为慢性肾炎，曾用中西药物而疗效不显。后来家长失去信心，不再予治疗。1976年12月底，患儿发热、咳嗽，以后出现嗜睡、鼻衄、恶心、呕吐、尿少，于1977年1月某日急诊入院。入院时体检:明显消瘦，皮肤干燥，鼻翼煽动，呼吸困难，心律不齐。实验室检查:二氧化碳结合力12.2容积%，尿素氮216mg%，血色素5.8g，诊断为慢性肾炎、尿毒症、酸中毒、继发性贫血。入院后立即输液、纠正酸中毒及脱水;予抗生素和中药真武汤、生脉散加味方，症状稍有稳定，二氧化碳结合力上升至56容积%，但全身症状无大改善，仍处于嗜睡衰竭状态，同时有鼻衄、呕吐咖啡样物。1月6日血色素降至4.5g，当时曾予输血。1月7日患儿情况转重，不能饮食，恶心呕吐频频发作，服药亦十分困难;大便1日数次，呈柏油样便;并有呕血、呼吸慢而不整(14~18次/分)，心率减至60~65次/分，当即予可拉明、洛贝林、生脉散注射液交替注射。1月8日，患儿继续呈嗜睡衰竭状态，面色晦暗，呼吸减慢，心率减慢至60次/分，大便仍为柏油便，情况越来越重，因急请会诊。诊查:会诊时，患儿呈嗜睡朦胧状态，时有恶心呕吐，呼吸深长而慢，脉沉细微弱无力而迟，舌嫩润齿痕尖微赤，苔薄白干中心微黄。
[核心病机]: 脾胃败绝;气阴两虚
[临证体会]: 患儿症状主要呈恶心呕吐，进食困难，嗜睡半朦胧状态，呕血便血，证属脾胃败绝之象。患儿呈嗜睡状，脉沉细无力而迟，舌嫩齿痕微赤中心薄黄，证属气阴两虚，结合患儿全身情况看应属气阴两竭，分析患儿发病全过程，肾病已久一直未愈，当前主要症状系继发于原有肾病的基础上;波及脾兼及心肺。
--------------
[临床资料]: 张某，男，38岁。初诊：1985年4月18日。主诉及病史：右耳因爆竹震聋，时历2个月。自感鸣响不息，耳边如有高声，则耳内倍觉不舒。西医诊断为“爆炸性耳聋”，经治未见好转。诊查：鼓膜完整，标志存在。音叉测验正常。舌薄苔，脉有弦意。
[核心病机]: 肝肺被伤
[临证体会]: 本例是由爆炸声引起的耳聋，中医认为，肝经循环于耳外，肺经结穴于耳中，且肝藏血、藏魂，肺主气、主魄。所以本案的发生是因惊魂骇魄，肝肺被伤，气滞血瘀所致。巨响惊魂骇魄，肝藏魂，肺藏魄，肝胆之经络环耳外，肺穴之笼葱在耳中，从此而鸣聋俱作，亦合乎情理之中。
--------------
[临床资料]: {clinical_info}
[核心病机]: {mechanism_str}
[临证体会]: 
"""
