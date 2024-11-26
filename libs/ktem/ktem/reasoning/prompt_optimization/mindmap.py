import logging

from ktem.llms.manager import llms

from kotaemon.base import BaseComponent, Document, HumanMessage, Node, SystemMessage
from kotaemon.llms import ChatLLM, PromptTemplate

logger = logging.getLogger(__name__)


class CreateMindmapPipeline(BaseComponent):
    """Create a mindmap from the question and context"""

    llm: ChatLLM = Node(default_callback=lambda _: llms.get_default())

    SYSTEM_PROMPT = """
        From now on you will behave as "MapGPT" and, for every text the user will submit, you are going to create a PlantUML mind map file for the inputted text to best describe main ideas. Format it as a code and remember that the mind map should be in the same language as the inputted context. You don't have to provide a general example for the mind map format before the user inputs the text.
    """  # noqa: E501
    MINDMAP_PROMPT_TEMPLATE = """
        Question:
        {question}

        Context:
        {context}

        Generate a sample PlantUML mindmap based on the provided question and context. Include only the information that is directly relevant to the question in the mindmap.

        Please note the following when generating the mindmap:
        1. Ensure the mindmap conforms to the PlantUML format, starting with `@startmindmap` and ending with `@endmindmap`.
        2. The title part should be presented directly without including "Title: ".
        3. Keep the content concise and to the point, removing any redundant information.

        Use the following template:

        @startmindmap
        * Main Topic
        ** Subtopic 1
        *** Detail A
        *** Detail B
        ** Subtopic 2
        *** Detail C
        @endmindmap

        Here is an example for you:

        @startmindmap
        * Climate Change & Environmental Protection
        ** Causes
        *** Greenhouse Gas Emissions
        **** Fossil Fuels
        **** Agriculture
        *** Deforestation
        ** Impacts
        *** Physical
        **** Sea Level Rise
        **** Extreme Weather
        *** Societal
        **** Food Security
        **** Health Risks
        ** Solutions
        *** Mitigation
        **** Renewable Energy
        **** Energy Efficiency
        **** Carbon Capture
        *** Adaptation
        **** Infrastructure
        **** Early Warning Systems
        ** International Efforts
        *** Paris Agreement
        *** Global Cooperation
        ** Public Role
        *** Awareness
        *** Sustainable Practices
        @endmindmap
    """  # noqa: E501
    prompt_template: str = MINDMAP_PROMPT_TEMPLATE

    def run(self, question: str, context: str) -> Document:  # type: ignore
        prompt_template = PromptTemplate(self.prompt_template)
        prompt = prompt_template.populate(
            question=question,
            context=context,
        )

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        return self.llm(messages)
