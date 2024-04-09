import re

from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

if "api_key_bool" not in st.session_state:
    st.session_state["api_key_bool"] = False

if "store_click" not in st.session_state:
    st.session_state["store_click"] = False

pattern = r'sk-.*'

llm = ChatOpenAI(
    temperature=0.1,
    api_key=st.session_state["api_key"] if st.session_state["api_key"] is not None else "_",
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_resource(show_spinner="Cloudflare 공식문서를 불러오고있습니다....")
def load_website(url):
    if (st.session_state["api_key_bool"] == True) and (st.session_state["api_key"] != None) and (
            re.match(pattern, st.session_state["api_key"])):
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
        )
        loader = SitemapLoader(
            url,
            filter_urls=(
                [
                    r"https:\/\/developers.cloudflare.com/ai-gateway.*",
                    r"https:\/\/developers.cloudflare.com/vectorize.*",
                    r"https:\/\/developers.cloudflare.com/workers-ai.*",
                ]
            ),
            parsing_function=parse_page,
        )
        loader.requests_per_second = 2
        docs = loader.load_and_split(text_splitter=splitter)
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(
            api_key=st.session_state["api_key"],
        ))
        return vector_store.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.markdown(
    """
    # SiteGPT

    Ask questions about the content of a cloudflare sitemap.
"""
)


def save_api_key(api_key):
    st.session_state["api_key"] = api_key
    st.session_state["api_key_bool"] = True


with st.sidebar:
    api_key = st.text_input("OPENAI_API_KEY를 넣어야 작동합니다.", disabled=st.session_state["api_key"] is not None).strip()

    if api_key:
        save_api_key(api_key)
        if not re.match(pattern, api_key):
            st.write("API_KEY가 올바르지 않습니다.")
        else:
            st.write("API_KEY가 올바르게 저장되었습니다.")

    button = st.button("저장")

    if button:
        save_api_key(api_key)
        st.session_state["store_click"] = True
        if api_key == "":
            st.write("API_KEY를 넣어주세요.")

    st.write("Made by Wonjang")
    st.write("https://github.com/wonjangcloud9/langchain/blob/main/pages/SiteGPT.py")

if (st.session_state["api_key_bool"]) and (st.session_state["api_key"] is not None) and (
        re.match(pattern, st.session_state["api_key"])) and (st.session_state["store_click"]):

    retriever = load_website("https://developers.cloudflare.com/sitemap.xml")
    query = st.text_input("Ask a question to the website.")
    if query:
        chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
        )
        result = chain.invoke(query)
        st.markdown(result.content.replace("$", "\$"))
