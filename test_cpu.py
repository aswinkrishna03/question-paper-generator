import question_generator
import app

pm = question_generator.get_paper_model("100")
blooms = question_generator.normalize_blooms_levels([])
concepts = ["test concept 1", "test concept 2", "test concept 3", "test concept 4", "test concept 5", "test concept 6"]
module_concepts = [
    {"module": 1, "concepts": [concepts[0]]},
    {"module": 2, "concepts": [concepts[1]]},
    {"module": 3, "concepts": [concepts[2]]},
    {"module": 4, "concepts": [concepts[3]]},
    {"module": 5, "concepts": [concepts[4], concepts[5]]},
]
generated = question_generator.generate_paper_fast(concepts, pm, blooms, module_concepts, "100")

print(generated)
print("Valid counts:", app._count_valid_generated_questions(generated))
print("100 marks valid?", app._is_valid_100_module_distribution(generated))
print("Illogical?", app._has_illogical_question_fragments(generated))
