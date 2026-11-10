from agent import Agent
from atomic_question_agent import AtomicQuestionAgent
from reranking_agent import RerankingAgent
from config import LLM_MODEL
import pandas


class Rag(Agent):
	def __init__(self, vector_db_path, corpus_df=None):
		super().__init__()
		self.atomic_question_agent_object = AtomicQuestionAgent()
		self.reranking_agent_object = RerankingAgent(vector_db_path, corpus_df=None)


	def ask_single_question(self, question):

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



	def ask_rag(self, user_interaction):
		print(f"Intraction utlisateur : {user_interaction}")

		list_of_questions = self.atomic_question_agent_object.split_question_into_list_of_atomic_questions(user_interaction)
		for question in list_of_questions:
			rag_response, documents, metadatas = self.ask_single_question(question=question)

			print(f"Pour la question : {question} - voici la réponse du RAG : {rag_response}")
			print("-"*20)
			print(f"Voici les sources :")
			for index_document in range(len(documents)):
				print(f"Documents {index_document}")
				print(documents[index_document])
				for key, value in metadatas[index_document].items():
					print(f"{key} : {value}")
				print("---")



if __name__ == "__main__":
	vector_db_path = "my_vector_db"
	corpus_df = pandas.read_csv("05_corpus_rag.csv")

	user_interaction = "Quelle est la couleur et le nom du chat de Bob ? C'est quoi Haute-Semelle ? C'est quoi une lampe à silence ? C'est quoi un livre oreiller ? C'est quoi un peigne-thermomètre"

	rag_object = Rag(vector_db_path=vector_db_path, corpus_df=corpus_df)
	rag_object.ask_rag(user_interaction=user_interaction)

