from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig
from deepeval.synthesizer import Evolution
from deepeval.synthesizer.config import EvolutionConfig
from deepeval.synthesizer.config import FiltrationConfig
from deepeval.synthesizer.config import ContextConstructionConfig

#Generator
styling_config = StylingConfig(
    scenario="RAG evaluator for extractive QA over provided context only",
    task=(
        "Generate only closed-book, context-grounded questions with one clear answer. "
        "No opinion, no hypothetical, no prediction, no comparative judgment unless explicitly stated in text. "
        "Prefer questions asking for exact names, roles, dates, counts, records, or direct stated facts."
    ),
    input_format=(
        "User asks factual question answerable from one passage. "
        "Question must be unambiguous and verifiable from the provided context."
    ),
    expected_output_format=(
        "One short factual answer, ideally 5-20 words. "
        "Must be directly supported by context wording. "
        "Do not add interpretation."
    )
)

evolution_config = EvolutionConfig(
    evolutions = {
        Evolution.MULTICONTEXT: 1/4, # sticks to the context
        Evolution.CONCRETIZING: 1/4, # sticks to the context
        Evolution.CONSTRAINED: 1/4, # sticks to the context
        Evolution.COMPARATIVE: 1/4, # sticks to the context
    },
    num_evolutions=1
)

filtration_config = FiltrationConfig(
    critic_model="gpt-4o-mini",
    synthetic_input_quality_threshold=0.5 
)

synthesizer = Synthesizer(
    model = "gpt-4o-mini",
    styling_config=styling_config,
    evolution_config = evolution_config,
    filtration_config = filtration_config
)

context_construction_config = ContextConstructionConfig(
    min_contexts_per_document=5,
)
goldens = synthesizer.generate_goldens_from_docs(
    document_paths = ["./docs/onepiece.txt"],
    include_expected_output=True,
    max_goldens_per_context=2,
    context_construction_config=context_construction_config
)


#Saving Dataset
dataframe = synthesizer.to_pandas()
print(dataframe)

directory = "./docs"
synthesizer.save_as(
    file_type = 'json',
    directory=directory,
    file_name="my_dataset"
)

