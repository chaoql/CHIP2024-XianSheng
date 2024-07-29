from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from typing import Optional, List, Mapping, Any, Sequence, Dict


class DummyNodePostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
            self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        # subtracts 1 from the score
        for n in nodes:
            n.score -= 1

        return nodes


import requests
from typing import List, Optional
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


class BgeRerank(BaseNodePostprocessor):
    url: str = Field(description="Rerank server url.")
    top_n: int = Field(description="Top N nodes to return.")

    def __init__(self, top_n: int, url: str):
        super().__init__(url=url, top_n=top_n)  # 调用TEI的rerank模型服务，实现rerank方法

    def rerank(self, query, texts):
        url = f"{self.url}/rerank"
        request_body = {"query": query, "texts": texts, "truncate": False}
        response = requests.post(url, json=request_body)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to rerank, detail: {response}")
        return response.json()

    @classmethod
    def class_name(cls) -> str:
        return "BgeRerank"  # 实现LlamaIndex要求的postprocessor的接口

    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[
        NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []  # 调用rerank
        texts = [node.text for node in nodes]
        results = self.rerank(query=query_bundle.query_str, texts=texts, )  # 组装返回nodes
        new_nodes = []
        for result in results[0: self.top_n]:
            new_node_with_score = NodeWithScore(node=nodes[int(result["index"])].node, score=result["score"], )
            new_nodes.append(new_node_with_score)
            return new_nodes
