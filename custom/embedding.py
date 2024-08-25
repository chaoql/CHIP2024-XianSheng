from typing import Any, List
from InstructorEmbedding import INSTRUCTOR

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from modelscope import snapshot_download


class InstructorEmbeddings(BaseEmbedding):
    _model: INSTRUCTOR = PrivateAttr()
    _instruction: str = PrivateAttr()

    def __init__(
            self,
            instructor_model_name: str = "iic/gte_Qwen2-7B-instruct",
            instruction: str = "",
            **kwargs: Any,
    ) -> None:
        # self._model = INSTRUCTOR(instructor_model_name)
        # model_dir = snapshot_download(instructor_model_name)
        # self._model =
        # self._instruction = instruction

        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, query]], prompt_name="query")
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, text]])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            [[self._instruction, text] for text in texts]
        )
        return embeddings
