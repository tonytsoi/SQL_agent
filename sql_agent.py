from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
import pandas as pd
import sqlite3
import streamlit as st
import credentials
import os
import uuid

os.environ['AWS_DEFAULT_REGION'] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key

thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

def generate_response(input_text):
    # Create the agent
    memory = MemorySaver()
    llm = init_chat_model("amazon.nova-pro-v1:0",
                            model_provider="bedrock_converse")

    connection = sqlite3.connect('./data/bike_store.db', check_same_thread=False)

    engine = create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
    )

    db = SQLDatabase(engine)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    system_prompt = """
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run,
    then look at the results of the query and return the answer. Unless the user
    specifies a specific number of examples they wish to obtain, always limit your
    query to at most {top_k} results.

    You can order the results by a relevant column to return the most interesting
    examples in the database. Never query for all the columns from a specific table,
    only ask for the relevant columns given the question.

    You MUST double check your query before executing it. If you get an error while
    executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
    database.

    To start you should ALWAYS look at the tables in the database to see what you
    can query. Do NOT skip this step.

    Then you should query the schema of the most relevant tables.
    """.format(
        dialect=db.dialect,
        top_k=5,
    )

    agent_executor = create_react_agent(llm, tools, prompt=system_prompt, checkpointer=memory)

    # Use the agent
    for step in agent_executor.stream(
            {"messages": [HumanMessage(content=f"{input_text}")]},
            config,
            stream_mode="values",
    ):
        # Print all outputs
        # yield step["messages"][-1]

        # Print only the relevant content
        if step["messages"][-1].type != 'human':
            try:
                for sentence in step["messages"][-1].content[0]['text'].split("/n"):
                    yield sentence + "  "
            except:
                for sentence in step["messages"][-1].content.split("/n"):
                    yield sentence.replace("```","") + "  "

    # Print only the last message
    # response = agent_executor.invoke(
    #         {"messages": [HumanMessage(content=f"{input_text}")]},
    #         config
    # )

    # yield response["messages"][-1].content
        
st.title("AI SQL Agent (Bike Store Data as Example")
st.caption("Powered by Amazon Nova Pro")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])
        
# Accept user input
if prompt := st.chat_input("What do you want to ask?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})