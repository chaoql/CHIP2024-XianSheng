from typing import List, Optional
from llama_index.core.schema import BaseNode
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
import Stemmer
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from custom.retriever import CustomRetriever
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
from llama_index.core import PromptTemplate

# from llama_index.postprocessor.jinaai_rerank import JinaRerank
choice_select_prompt ="""文件列表如下。每份文件旁边都有一个编号和文件摘要。还提供了一句描述。
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

# choice_select_prompt = ("A list of documents is shown below. Each document has a number next to it along "
#                         "with a summary of the document. A question is also provided. \n"
#                         "Respond with the numbers of the documents "
#                         "you should consult to answer the question, in order of relevance, as well \n"
#                         "as the relevance score. The relevance score is a number from 1-10 based on "
#                         "how relevant you think the document is to the question.\n"
#                         "Do not include any documents that are not relevant to the question. \n"
#                         "Example format: \n"
#                         "Document 1:\n<summary of document 1>\n\n"
#                         "Document 2:\n<summary of document 2>\n\n"
#                         "...\n\n"
#                         "Document 10:\n<summary of document 10>\n\n"
#                         "Question: <question>\n"
#                         "Answer:\n"
#                         "Doc: 9, Relevance: 7\n"
#                         "Doc: 3, Relevance: 4\n"
#                         "Doc: 7, Relevance: 3\n\n"
#                         "Answers should be consistent with the Example format. Let's try this now: \n\n"
#                         "{context_str}\n"
#                         "Question: {query_str}\n"
#                         "Answer:\n")

choice_select_prompt = PromptTemplate(choice_select_prompt)


def build_query_engine(with_LLMrerank, rerank_top_k, hybrid_mode,
                       index: VectorStoreIndex,
                       response_mode: ResponseMode = ResponseMode.TREE_SUMMARIZE,
                       qa_prompt_tmpl: Optional[BasePromptTemplate] = None,
                       hybrid_search: bool = False,
                       top_k: int = 2,
                       nodes: Optional[List[BaseNode]] = None):
    if hybrid_search:
        if with_LLMrerank:
            rag_query_engine = index.as_query_engine(similarity_top_k=top_k,
                                                     text_qa_template=qa_prompt_tmpl,
                                                     node_postprocessors=[
                                                         LLMRerank(top_n=rerank_top_k, llm=Settings.llm,
                                                                   choice_select_prompt=choice_select_prompt)],
                                                     sparse_top_k=12,
                                                     vector_store_query_mode="hybrid",
                                                     # response_synthesizer=get_response_synthesizer(
                                                     #     response_mode=response_mode),
                                                     )
        else:
            rag_query_engine = index.as_query_engine(similarity_top_k=top_k,
                                                     text_qa_template=qa_prompt_tmpl,
                                                     node_postprocessors=[],
                                                     sparse_top_k=12,
                                                     vector_store_query_mode="hybrid",
                                                     # response_synthesizer=get_response_synthesizer(
                                                     #     response_mode=response_mode),
                                                     )
    else:
        if with_LLMrerank:
            # Build a tree index over the set of candidate nodes, with a summary prompt seeded with the query. with LLM reranker
            rag_query_engine = index.as_query_engine(similarity_top_k=top_k,
                                                     text_qa_template=qa_prompt_tmpl,
                                                     node_postprocessors=[
                                                         LLMRerank(top_n=rerank_top_k, llm=Settings.llm,
                                                                   choice_select_prompt=choice_select_prompt)],
                                                     # response_synthesizer=get_response_synthesizer(
                                                     #     response_mode=response_mode),
                                                     )
        else:
            rag_query_engine = index.as_query_engine(similarity_top_k=top_k,
                                                     text_qa_template=qa_prompt_tmpl,
                                                     node_postprocessors=[],
                                                     # response_synthesizer=get_response_synthesizer(
                                                     #     response_mode=response_mode),
                                                     )
    return rag_query_engine
