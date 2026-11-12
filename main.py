from rag import Rag
import os

if __name__ == "__main__":
	vector_db_path = "my_vector_db"
	corpus_path = "05_corpus_rag.csv"

	
	user_interaction = input("Posez une question au RAG :\n")

	rag_object = Rag(vector_db_path=vector_db_path, corpus_path=corpus_path)
	dict_of_questions_answers = rag_object.ask_rag(user_interaction=user_interaction, verbose=False)

	os.system("clear")
	for question, value in dict_of_questions_answers.items():
		print(value['rag_response'])