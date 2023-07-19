from llama_cpp import Llama
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain import PromptTemplate, LLMChain
from typing import Any, List, Mapping, Optional
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory


class CustomLLM(LLM):
    model: Any

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        return self.model(prompt, temperature=0.7, top_p=0.9,
                          top_k=20, repeat_penalty=1.15, mirostat_mode=0, mirostat_tau=5, mirostat_eta=0.1)["choices"][0]["text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""

        return {"model": self.model}


prompt = PromptTemplate(
    input_variables=["query"],
    template="""You are a helpful AI assistant, you will answer the users query
with a short but precise answer. If you are not sure about the answer you state
"I don't know". This is a conversation, not a webpage, there should be ZERO HTML
in the response.

Remember, Assistant responses are short. Please do not stop generating mid sentence. Here is the conversation:

User: {query}
Assistant: """
)


question = "Explain the difference between nuclear fission and fussion"
model = Llama(model_path="./ggml-model-q4_0.bin", n_ctx=2048, seed=0, n_threads=None,
              n_batch=512, use_mmap=True, use_mlock=False, low_vram=False, n_gpu_layers=1, verbose=False)
llm = CustomLLM(model=model)
# llm_chain = LLMChain(prompt=prompt, llm=customllm)
# print(llm_chain.run(question))
conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)
while True:
    message = input("User: ")
    print("Assistent: "+conversation_buf(message)["response"])
