from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool

def search_wikipedia_full_content(query):
    """Get full Wikipedia content, not just summary"""
    api_wrapper = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=1000,
        load_all_available_meta=True
    )
    
    # Use load() instead of run() to get full content
    documents = api_wrapper.load(query)
    
    if not documents:
        return "No Wikipedia page found for this query."
    
    doc = documents[0]
    
    # The page_content contains the full article content
    full_content = doc.page_content
    metadata = doc.metadata
    
    return f"Title: {metadata.get('title', 'Unknown')}\nURL: {metadata.get('source', 'Unknown')}\n\nFull Content:\n{full_content}"

wikipedia_tool = Tool(
    name="search_wikipedia",
    func=search_wikipedia_full_content,
    description="Searches Wikipedia for the given query and returns the full content"
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search_web",
    func=search.run,
    description="Searches the web for the given query and returns the results"
)