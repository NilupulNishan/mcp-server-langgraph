import asyncio
import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
  


load_dotenv()

async def main():
    openai_key=os.getenv("AZURE_OPENAI_API_KEY")
    owm_key=os.getenv("OWM_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in .env or environment variables.")
    if not owm_key:
        raise ValueError("OWM_API_KEY not found in .env or environment variables.")
    

    model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_GPT4O_MINI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
    )

    # client  = MultiServerMCPClient(
    #     {
    #         "math": {
    #             "command": "python",
    #             "args": [r"C:/Users/nilup/gitrepos/PERSONAL/mcp-server-langgraph/custom-mcp-server/custom_mcp_server.py"],
    #             "transport": "stdio",
    #         }
    #     }
    # )

    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "streamable-http",
                "url": "http://127.0.0.1:8000/mcp"  # URL of your MCP server
            }
        }
    ) 

    tools = await client.get_tools()

    model_with_tools = model.bind_tools(tools)

    tool_node = ToolNode(tools)

    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
    
    async def call_model(state: MessagesState):
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}
    
    builder = StateGraph(MessagesState)
    
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        should_continue,
    )
    builder.add_edge("tools", "call_model")

    # Compile the graph
    graph = builder.compile()

    # running the graph
    result = await graph.ainvoke({"messages": "what's (3 + 5) x 12?"})
    print(result["messages"][-1].content)   

if __name__ == "__main__":
    asyncio.run(main())
