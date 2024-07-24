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

from custom.query import build_query_engine
from custom.prompt import qa_prompt_tmpl_str, simple_qa_prompt_tmpl_str
from llama_index.readers.json import JSONReader

warnings.filterwarnings('ignore')


def load_data(input_file, persist_dir):
    reader = JSONReader(
        # The number of levels to go back in the JSON tree. Set to 0 to traverse all levels. Default is None.
        levels_back=0,
        # # The maximum number of characters a JSON fragment would be collapsed in the output. Default is None.
        # collapse_length="<Collapse Length>",
        # # If True, ensures that the output is ASCII-encoded. Default is False.
        # ensure_ascii="<Ensure ASCII>",
        # # If True, indicates that the file is in JSONL (JSON Lines) format. Default is False.
        # is_jsonl="<Is JSONL>",
        # # If True, removes lines containing only formatting from the output. Default is True.
        # clean_json="<Clean JSON>",
    )

    # Load data from JSON file
    documents = reader.load_data(input_file=input_file, extra_info={})

    # Sliding windows chunking & Extract nodes from documents
    node_parser = SentenceWindowNodeParser.from_defaults(
        # how many sentences on either side to capture
        window_size=3,
        # the metadata key that holds the window of surrounding sentences
        window_metadata_key="window",
        # the metadata key that holds the original sentence
        original_text_metadata_key="original_sentence",
    )
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)

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


def Build_query_engine(index, top_k, hybrid_search, nodes, with_hyde):
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    simple_qa_prompt_tmpl = PromptTemplate(simple_qa_prompt_tmpl_str)  # norag
    rag_query_engine = build_query_engine(index, ResponseMode.TREE_SUMMARIZE, qa_prompt_tmpl, hybrid_search, top_k,
                                          nodes)
    simple_query_engine = index.as_query_engine(similarity_top_k=top_k,
                                                text_qa_template=simple_qa_prompt_tmpl,
                                                response_synthesizer=get_response_synthesizer(
                                                    response_mode=ResponseMode.GENERATION),
                                                )

    # HyDE(当问题较为简单时，不需要该模块参与)
    if with_hyde:
        hyde = HyDEQueryTransform(include_original=True)
        rag_query_engine = TransformQueryEngine(rag_query_engine, hyde)

    # # Router Query Engine(Query Classification)
    # rag_tool = QueryEngineTool.from_defaults(
    #     query_engine=rag_query_engine,
    #     description=rag_description,
    # )
    # simple_tool = QueryEngineTool.from_defaults(
    #     query_engine=simple_query_engine,
    #     description=norag_rag_description,
    # )
    # query_engine = RouterQueryEngine(
    #     selector=LLMSingleSelector.from_defaults(),
    #     query_engine_tools=[
    #         rag_tool,
    #         simple_tool,
    #     ],
    # )
    return rag_query_engine, simple_query_engine
