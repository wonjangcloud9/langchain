import json
import re

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

pattern = r'sk-.*'

if "difficult" not in st.session_state:
    st.session_state["difficult"] = "easy"

if "score" not in st.session_state:
    st.session_state["score"] = 0

if "is_finished" not in st.session_state:
    st.session_state["is_finished"] = False

if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)


def on_press_easy():
    st.session_state['difficult'] = "easy"


def on_press_medium():
    st.session_state['difficult'] = "medium"


def on_press_hard():
    st.session_state['difficult'] = "hard"


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_resource(show_spinner="Making quiz...")
def run_quiz_chain(_docs):
    chain = prompt | llm
    return chain.invoke({
        "topic": topic,
        "difficulty": st.session_state["difficult"],
    })


@st.cache_resource(show_spinner="Search Wiki...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5, lang="ko")
    docs = retriever.get_relevant_documents(term)
    return docs


api_key = st.sidebar.text_input("Enter your OpenAI API Key", )

if api_key:
    st.session_state["api_key"] = api_key
    llm = ChatOpenAI(
        openai_api_key=st.session_state["api_key"] if st.session_state["api_key"] is not None else "_",
        model="gpt-3.5-turbo-0125",
        temperature=0.1,
    ).bind(
        function_call="auto",
        functions=[
            function,
        ],
    )
    easy = st.sidebar.button("Easy", key="easy", on_click=on_press_easy)
    medium = st.sidebar.button("Medium", key="medium", on_click=on_press_medium)
    hard = st.sidebar.button("Hard", key="hard", on_click=on_press_hard)

    st.title("QuizGPT")

    docs = None

    prompt = PromptTemplate.from_template(
        """            
        Topic: {topic}에 해당하는 주제에 대한 문제를 만들어주세요.
        3 개의 문제를 만들어주세요.
        난이도: {difficulty}에 의해 결정됩니다.
        질문의 답변은 4가지 문항에서 선택할 수 있습니다.
        1개의 정답과 3개의 오답이 있어야 합니다.
        """,
    )

    with st.sidebar:
        st.write(f"{st.session_state['difficult']}")
        if st.session_state["api_key"] is not None:
            if "api_key" in st.session_state and re.match(pattern, st.session_state["api_key"]):
                topic = st.text_input("Search Wikipedia...")
                if topic:
                    docs = wiki_search(topic)

    if docs:

        response = run_quiz_chain(docs)
        response = response.additional_kwargs["function_call"]["arguments"]
        response = json.loads(response)

        with st.form("quiz_questions_form"):
            for index, question in enumerate(response["questions"]):
                st.write(f"Question {index + 1}")
                st.write(question["question"])
                option = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    label_visibility="collapsed",
                    key=f"question_{index}"
                )
                if {"answer": option, "correct": True} in question["answers"]:
                    st.session_state["score"] += 1
            submitted = st.form_submit_button("Submit")

            if submitted:
                print(st.session_state["score"])
                st.session_state["is_finished"] = True
                if st.session_state["score"] == 3:
                    st.balloons()
                    st.write("Thank you for playing!")
                    st.session_state["score"] = 0
                else:
                    st.write(f"Your score is {st.session_state['score']}")
                    st.session_state["score"] = 0
                    st.session_state["is_finished"] = False

with st.sidebar:
    st.write("Made by Wonjang")
    st.write("https://github.com/wonjangcloud9/langchain/blob/main/pages/QuizGPT.py")
