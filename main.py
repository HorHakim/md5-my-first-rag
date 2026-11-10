from rag import Rag 
from atomic_question_agent import AtomicQuestionAgent
import pandas

class RagApp:
	def __init__(self, vector_db_path, corpus_df=None):
		self.atomic_question_agent_object = AtomicQuestionAgent()
		self.rag_object = Rag(vector_db_path, corpus_df)


	def ask_rag_app(self, user_interaction):
		print(f"Intraction utlisateur : {user_interaction}")

		list_of_questions = self.atomic_question_agent_object.split_question_into_list_of_atomic_questions(user_interaction)
		for question in list_of_questions:
			rag_response, documents, metadatas = self.rag_object.ask_rag(question=question)

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
	rag_app_object = RagApp(vector_db_path=vector_db_path, corpus_df=corpus_df)
	user_interaction = "Quelle est la couleur et le nom du chat de Bob ? C'est quoi Haute-Semelle ? C'est quoi une lampe à silence ? C'est quoi un livre oreiller ? C'est quoi un peigne-thermomètre"
	rag_app_object.ask_rag_app(user_interaction=user_interaction)