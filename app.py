import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","Act like a helpful AI assistant and respond to user queries"),
        ("user","Query:{query}")
    ]
)


def generate_response(query, api_key, engine, temperature, max_tokens):
    llm = ChatOpenAI(
        model=engine,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'query': query})
    return answer



st.title("Q&A Chatbot With OpenAI")

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your OpenAI API Key:",type="password")

## Select the OpenAI model
engine=st.sidebar.selectbox("Select Open AI model",["gpt-4o","gpt-4-turbo","gpt-4"])

## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Ask any question")
user_input=st.text_input("You:")

if user_input and api_key:
    response=generate_response(user_input,api_key,engine,temperature,max_tokens)
    st.write(response)

elif user_input:
    st.warning("Please enter your OpenAI API Key")
