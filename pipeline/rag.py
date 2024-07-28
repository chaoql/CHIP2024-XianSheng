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

warnings.filterwarnings('ignore')


def load_data(input_file, persist_dir):
    documents = SimpleDirectoryReader(input_files=[input_file]).load_data()
    node_parser = JSONNodeParser()
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)

    # 创建一个持久化的索引到磁盘
    client = QdrantClient(path=persist_dir)
    # 创建启用混合索引的向量存储
    vector_store = QdrantVectorStore(
        "test", client=client, enable_hybrid=True, batch_size=20
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    Settings.chunk_size = 512
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    # indexing & storing
    try:
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=persist_dir),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir),
        )
        index = load_index_from_storage(storage_context)
    except:
        index = VectorStoreIndex(nodes=nodes)
        index.storage_context.persist(persist_dir=persist_dir)
    return index, nodes


def Build_query_engine(rerank_top_k, hybrid_mode, index, top_k, hybrid_search, nodes, with_hyde, qa_prompt_tmpl):
    # qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl)
    rag_query_engine = build_query_engine(rerank_top_k, hybrid_mode, index, ResponseMode.TREE_SUMMARIZE, qa_prompt_tmpl,
                                          hybrid_search, top_k, nodes)

    # HyDE(当问题较为简单时，不需要该模块参与)
    if with_hyde:
        hyde = HyDEQueryTransform(include_original=True)
        rag_query_engine = TransformQueryEngine(rag_query_engine, hyde)

    return rag_query_engine
