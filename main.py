from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import wikipedia_tool, search_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]

llm = ChatGroq(model="llama-3.3-70b-versatile")

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that helps users find information and summarize research topics.

            Use the available tools to research the topic:
            1. search_wikipedia - for encyclopedic information
            2. search_web - for general web search

            After gathering the required information, provide a comprehensive summary with:
            {format_instructions}
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{user_input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [wikipedia_tool, search_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

user_input = input("Provide your research topic: ")

raw_response = agent_executor.invoke(
    {
        "user_input": user_input
    }
)
try:
    structured_response = parser.parse(raw_response.get("output"))

    print(structured_response)

    print("Topic: ", structured_response.topic)
    print("Summary: ", structured_response.summary)
    print("Source: ", structured_response.source)
    print("Tools Used: ", structured_response.tools_used)
except Exception as e:
    print("Error occurred while parsing response:", e)
