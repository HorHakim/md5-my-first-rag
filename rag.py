from agent import Agent
from reranking_agent import RerankingAgent
from config import LLM_MODEL
import pandas


class Rag(Agent):
	def __init__(self, vector_db_path, corpus_df=None):
		super().__init__()
		self.reranking_agent_object = RerankingAgent(vector_db_path, corpus_df=None)


	def ask_rag(self, question):

		prompt_system, usefull_documents, usefull_metadatas = self.reranking_agent_object.build_rag_context(question)
		
		chat_completion = self.client.chat.completions.create(
			messages=[
				{
					"role": "system",
					"content": prompt_system
				},
				{
					"role": "user",
					"content": question,
				}
			],
			temperature=0,
			model=LLM_MODEL
		)

		rag_response = chat_completion.choices[0].message.content
		return rag_response, usefull_documents, usefull_metadatas


if __name__ == "__main__":
	vector_db_path = "my_vector_db"
	corpus_df = pandas.read_csv("05_corpus_rag.csv")
	rag_object = Rag(vector_db_path=vector_db_path, corpus_df=corpus_df)

	rag_response, usefull_documents, usefull_metadatas = rag_object.ask_rag(question="Quelle est la couleur et le nom du chat de Bob ?")

	print(rag_response)
	print("-"*20)
	for index_document in range(len(usefull_documents)):
		print(f"usefull_Documents {index_document}")
		print(usefull_documents[index_document])
		for key, value in usefull_metadatas[index_document].items():
			print(f"{key} : {value}")
		print("---")