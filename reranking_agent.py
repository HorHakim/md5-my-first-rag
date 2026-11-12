from agent import Agent
from config import LLM_MODEL
from vector_db import VectorDB

import json

class RerankingAgent(Agent):
	def __init__(self, vector_db_path, corpus_path=None):
		super().__init__()
		self.vector_db_object = VectorDB(vector_db_path=vector_db_path, corpus_path=corpus_path)


	def build_rag_context(self, question):
		documents, metadatas = self.vector_db_object.retrieve(question)

		usefull_documents, usefull_metadatas = self.filter_sources(question, documents, metadatas)

		prompt_system = RerankingAgent.read_file("./prompts_system/rag.txt")

		prompt_system = prompt_system.replace("{{CHUNKS}}", "\n\t".join(usefull_documents))

		return prompt_system, usefull_documents, usefull_metadatas


	def filter_sources(self, question, documents, metadatas):
		
		chat_completion = self.client.chat.completions.create(
			messages=[
				{
					"role": "system",
					"content": RerankingAgent.read_file(file_path="./prompts_system/reranking_agent.txt")
				},
				{
					"role": "user",
					"content": f"Question: {question}\nDocuments: {documents}\nMetadatas : {metadatas}",
				}
			],
			temperature=0,
			model=LLM_MODEL, 
			response_format={"type": "json_object"},
			max_completion_tokens=10_000
		)

		reraking_agent_response = json.loads(chat_completion.choices[0].message.content)

		usefull_documents = reraking_agent_response["usefull_documents"]
		usefull_metadatas = reraking_agent_response["usefull_metadatas"]

		return usefull_documents, usefull_metadatas
