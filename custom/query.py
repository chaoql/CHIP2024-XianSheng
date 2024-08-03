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
from custom.prompt import choice_select_prompt_str

# from llama_index.postprocessor.jinaai_rerank import JinaRerank


choice_select_prompt = PromptTemplate(choice_select_prompt_str)


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
                                                     #     response_mode=response_mode,
                                                     #     refine_template=PromptTemplate(refine_tmpl_str)),
                                                     )
        else:
            rag_query_engine = index.as_query_engine(similarity_top_k=top_k,
                                                     text_qa_template=qa_prompt_tmpl,
                                                     node_postprocessors=[],
                                                     sparse_top_k=12,
                                                     vector_store_query_mode="hybrid",
                                                     # response_synthesizer=get_response_synthesizer(
                                                     #     response_mode=response_mode,
                                                     #     refine_template=PromptTemplate(refine_tmpl_str)),
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
                                                     #     response_mode=response_mode,
                                                     #     refine_template=PromptTemplate(refine_tmpl_str)),
                                                     )
        else:
            rag_query_engine = index.as_query_engine(similarity_top_k=top_k,
                                                     text_qa_template=qa_prompt_tmpl,
                                                     node_postprocessors=[],
                                                     # response_synthesizer=get_response_synthesizer(
                                                     #     response_mode=response_mode,
                                                     #     refine_template=PromptTemplate(refine_tmpl_str)),
                                                     )
    return rag_query_engine
