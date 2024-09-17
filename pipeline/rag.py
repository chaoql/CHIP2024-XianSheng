from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import PromptTemplate, get_response_synthesizer, StorageContext, VectorStoreIndex, \
    SimpleDirectoryReader, Settings
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core import load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
import warnings
from custom.glmfz import ChatGLM
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.node_parser import JSONNodeParser
from custom.query import build_query_engine
from llama_index.readers.json import JSONReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core import load_index_from_storage

warnings.filterwarnings('ignore')


def load_json_text_hybrid_data(json_files, text_files, persist_dir):
    documents = SimpleDirectoryReader(input_files=json_files).load_data()
    node_parser = JSONNodeParser()
    nodes_1 = node_parser.get_nodes_from_documents(documents, show_progress=True)

    documents = SimpleDirectoryReader(input_files=text_files).load_data()
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    nodes_2 = node_parser.get_nodes_from_documents(documents, show_progress=True)
    nodes = nodes_1 + nodes_2

    # 创建一个持久化的索引到磁盘
    client = QdrantClient(path=persist_dir)
    # 创建启用混合索引的向量存储
    vector_store = QdrantVectorStore(
        "test", client=client, enable_hybrid=True, batch_size=20
    )
    try:
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
        index = load_index_from_storage(storage_context, show_progress=True)
    except:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        Settings.chunk_size = 512
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        index.storage_context.persist(persist_dir=persist_dir)
    return index, nodes


def load_json_text_data(json_files, text_files, persist_dir):
    documents = SimpleDirectoryReader(input_files=json_files).load_data()
    # documents = SimpleDirectoryReader(input_files=["data/trainTask4.json"]).load_data()
    node_parser = JSONNodeParser()
    nodes4_1 = node_parser.get_nodes_from_documents(documents, show_progress=True)
    documents = SimpleDirectoryReader(input_files=text_files).load_data()
    # documents = SimpleDirectoryReader(
    #     input_files=["extra_data/doctor/中医诊断学2.txt", "extra_data/doctor/中医内科学.txt",
    #                  "extra_data/doctor/中医基础理论.txt", ]).load_data()
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    nodes4_2 = node_parser.get_nodes_from_documents(documents, show_progress=True)
    nodes4 = nodes4_1 + nodes4_2
    # indexing & storing
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index4 = load_index_from_storage(storage_context, show_progress=True)
    except:
        index4 = VectorStoreIndex(nodes=nodes4, show_progress=True)
        index4.storage_context.persist(persist_dir=persist_dir)
    return index4, nodes4

def load_all_data(input_dir, persist_dir):
    documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

    # indexing & storing
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context, show_progress=True)
    except:
        index = VectorStoreIndex(nodes=nodes, show_progress=True)
        index.storage_context.persist(persist_dir=persist_dir)
    return index, nodes


def load_txt_data(input_file, persist_dir, chunk_size=512, chunk_overlap=128):
    documents = SimpleDirectoryReader(input_files=input_file).load_data()
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=128)
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

    # indexing & storing
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context, show_progress=True)
    except:
        index = VectorStoreIndex(nodes=nodes, show_progress=True)
        index.storage_context.persist(persist_dir=persist_dir)
    return index, nodes


def load_data(input_file, persist_dir):
    documents = SimpleDirectoryReader(input_files=input_file).load_data()
    node_parser = JSONNodeParser()
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

    # indexing & storing
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context, show_progress=True)
    except:
        index = VectorStoreIndex(nodes=nodes, show_progress=True)
        index.storage_context.persist(persist_dir=persist_dir)
    return index, nodes


def load_hybrid_data(input_file, persist_dir):
    documents = SimpleDirectoryReader(input_files=input_file).load_data()
    node_parser = JSONNodeParser()
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

    # 创建一个持久化的索引到磁盘
    client = QdrantClient(path=persist_dir)
    # 创建启用混合索引的向量存储
    vector_store = QdrantVectorStore(
        "test", client=client, enable_hybrid=True, batch_size=20
    )
    try:
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
        index = load_index_from_storage(storage_context, show_progress=True)
    except:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        Settings.chunk_size = 512
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        index.storage_context.persist(persist_dir=persist_dir)
    return index, nodes


def Build_query_engine(with_LLMrerank, rerank_top_k, index, top_k, response_mode, hybrid_search, nodes, with_hyde,
                       qa_prompt_tmpl):
    rag_query_engine = build_query_engine(with_LLMrerank, rerank_top_k, index, response_mode,
                                          qa_prompt_tmpl, hybrid_search, top_k, nodes)

    # HyDE(当问题较为简单时，不需要该模块参与)
    if with_hyde:
        hyde = HyDEQueryTransform(include_original=True)
        rag_query_engine = TransformQueryEngine(rag_query_engine, hyde)

    return rag_query_engine
