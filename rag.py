from agent import Agent
from moderator_agent import ModeratorAgent
from atomic_question_agent import AtomicQuestionAgent
from reranking_agent import RerankingAgent
from config import LLM_MODEL
import os


class Rag(Agent):
	def __init__(self, vector_db_path, corpus_path=""):
		super().__init__()
		self.moderator_agent_object = ModeratorAgent()
		self.atomic_question_agent_object = AtomicQuestionAgent()
		self.reranking_agent_object = RerankingAgent(vector_db_path, corpus_path)


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



	def ask_rag(self, user_interaction, verbose=False):
		is_legitimate, justification = self.moderator_agent_object.moderate_user_interaction(user_interaction)
		if not is_legitimate:
			raise(Exception(f"Prompt Injection : {justification}"))

		list_of_questions = self.atomic_question_agent_object.split_question_into_list_of_atomic_questions(user_interaction)

		dict_of_questions_answers = dict()

		for question in list_of_questions:
			rag_response, documents, metadatas = self.ask_single_question(question=question)

			answer = dict()
			answer["rag_response"] = rag_response
			answer["documents"] = documents
			answer["metadatas"] = metadatas

			dict_of_questions_answers[question] = answer

		if verbose:
			Rag.show_answers(dict_of_questions_answers)

		return dict_of_questions_answers


	@staticmethod
	def show_answers(dict_of_questions_answers):
		os.system("clear")

		print(f"Intraction utlisateur : {user_interaction}")


		for question, answer in dict_of_questions_answers.items():
			print(f"Pour la question : {question}\n")
			print(f"Voici la réponse du RAG : {answer["rag_response"]}")
			print("-"*20)
			print(f"Voici les sources :")
			for document, metadata in zip(answer["documents"], answer["metadatas"]):
				print(document)
				print(f"Metadatas :")
				for key, value in metadata.items():
					print(f"\t{key} : {value}")
				print("---")
			print("-"*80)



if __name__ == "__main__":
	vector_db_path = "my_vector_db"
	corpus_path = "05_corpus_rag.csv"


	rag_object = Rag(vector_db_path=vector_db_path, corpus_path=corpus_path)
	user_interaction = "Quelle est la couleur et le nom du chat de Bob ? C'est quoi Haute-Semelle ? C'est quoi une lampe à silence ? C'est quoi un livre oreiller ? C'est quoi un peigne-thermomètre"
	rag_object.ask_rag(user_interaction=user_interaction, verbose=True)
	




