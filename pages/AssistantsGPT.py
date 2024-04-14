import json
import time
import os

import streamlit as st

from langchain.utilities import DuckDuckGoSearchAPIWrapper

from openai import OpenAI

if "messages" not in st.session_state:
    st.session_state["messages"] = []


class DiscussionClient:

    def __init__(self):
        pass

    def save_message(self, message, role):
        st.session_state["messages"].append({"message": message, "role": role})

    def send_message(self, message, role, save=True):
        with st.chat_message(role):
            st.markdown(message)

        if save:
            self.save_message(message, role)

    def paint_history(self):
        for message in st.session_state["messages"]:
            self.send_message(message["message"], message["role"], save=False)


class ThreadClient:
    def __init__(self, client):
        self.client = client

    def get_run(self, run_id, thread_id):
        return self.client.beta.threads.runs.retrieve(
            run_id=run_id,
            thread_id=thread_id,
        )

    def send_message(self, thread_id, content):
        return self.client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=content
        )

    def get_messages(self, thread_id):
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        messages = list(messages)
        messages.reverse()
        for message in messages:
            if message.role == "user":
                discussion_client.send_message(message.content[0].text.value, "user")

    def get_tool_outputs(self, run_id, thread_id):
        run = self.get_run(run_id, thread_id)
        outputs = []
        for action in run.required_action.submit_tool_outputs.tool_calls:
            action_id = action.id
            function = action.function
            outputs.append({
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            })
        return outputs

    def submit_tool_outputs(self, run_id, thread_id):
        outputs = self.get_tool_outputs(run_id, thread_id)
        discussion_client.send_message("이슈를 찾았어요!", "ai")
        discussion_client.send_message(outputs[0]["output"], "ai")

        return self.client.beta.threads.runs.submit_tool_outputs(
            run_id=run_id, thread_id=thread_id, tool_outputs=outputs
        )

    def wait_on_run(self, run, thread):
        while run.status == "queued" or run.status == "in_progress":
            run = self.get_run(run.id, thread.id)
            time.sleep(0.5)
        return run


class IssueSearchClient:

    def __init__(self):
        self.ddg = DuckDuckGoSearchAPIWrapper()

    def get_issue(self, category_data):
        category_data = category_data.get("category", "")

        return self.ddg.run(category_data)


issue_search_client = IssueSearchClient()
discussion_client = DiscussionClient()

functions_map = {
    "get_issue": issue_search_client.get_issue,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_issue",
            "description": "최신 이슈를 한글로 알려줍니다. 정리도 좀 해가지고 말이죠.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "최신 이슈를 한글로 알려줍니다. 정리도 좀 해가지고 말이죠."
                    }
                },
                "required": ["category"],
            },
        },
    },
]

st.set_page_config(
    page_title="AssistantsGPT",
    page_icon="🤖",
)

st.markdown(
    """
    
    # AssistantsGPT
        
    :rainbow: :rainbow[안녕하세요!!] :rainbow:  
    
    저는 GPT4로 만들어진 이슈왕입니다. :dizzy:
    
    이슈를 빠르게 찾아드릴게요. :sparkles:
    
    저에게 물어보세요! :tulip: :tulip: :tulip:
    
    """
)

api_key = st.sidebar.text_input("당신의 OpenAI API Key를 입력하세요")

if api_key and api_key.startswith("sk-"):
    st.session_state["api_key"] = api_key
    client = OpenAI(api_key=api_key)

    assistant_id = "asst_5Sdl0TQ8xVaq7hQAxVAsFAy1"

    category = st.text_input("대화를 시작하세요.")

    if category:
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": f"I want to know {category}",
                }
            ]
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,

        )

        assistant = ThreadClient(client)
        run = assistant.wait_on_run(run, thread)

        if run:
            discussion_client.send_message("이슈를 찾고 있어요!", "ai", save=False)
            discussion_client.paint_history()
            assistant.get_messages(thread.id)
            assistant.submit_tool_outputs(run.id, thread.id)
            st.download_button(
                label="채팅 내역 다운로드",
                data=json.dumps(st.session_state["messages"]),
                file_name="chat_history.txt",
                mime="text/plain",
            )

with st.sidebar:
    st.write("Made by Wonjang")
    st.write("https://github.com/wonjangcloud9/langchain/blob/main/pages/AssistantsGPT.py")
