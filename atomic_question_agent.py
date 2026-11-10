from agent import Agent
from config import LLM_MODEL

import json

class AtomicQuestionAgent(Agent):
	def __init__(self):
		super().__init__()


	def split_question_into_list_of_atomic_questions(self, user_interaction):
		chat_completion = self.client.chat.completions.create(
			messages=[
				{
					"role": "system",
					"content": AtomicQuestionAgent.read_file(file_path="./atomic_question_agent_prompt_system.txt")
				},
				{
					"role": "user",
					"content": user_interaction,
				}
			],
			temperature=0,
			model=LLM_MODEL, 
			response_format={"type": "json_object"},
		)

		list_of_questions = json.loads(chat_completion.choices[0].message.content)["list_of_questions"]
		return list_of_questions



if __name__ == "__main__":
	atomic_question_agent_object = AtomicQuestionAgent()
	user_interaction = "Quelle est la couleur et le nom du chat de Bob ? C'est quoi Haute-Semelle ? C'est quoi une lampe à silence ? C'est quoi un livre oreiller ? C'est quoi un peigne-thermomètre"
	list_of_questions = atomic_question_agent_object.split_question_into_list_of_atomic_questions(user_interaction=user_interaction)
	print(list_of_questions)