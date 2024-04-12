import json
import time

import streamlit as st

from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from openai import OpenAI


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
            print(f"{message.role}: {message.content[0].text.value}")

    def get_tool_outputs(self, run_id, thread_id):
        run = self.get_run(run_id, thread_id)
        outputs = []
        for action in run.required_action.submit_tool_outputs.tool_calls:
            action_id = action.id
            function = action.function
            print(f"Calling function: {function.name} with arg {function.arguments}")
            outputs.append({
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            })
        return outputs

    def submit_tool_outputs(self, run_id, thread_id):
        outputs = self.get_tool_outputs(run_id, thread_id)
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
        self.wiki = WikipediaAPIWrapper()
        self.ddg = DuckDuckGoSearchAPIWrapper()

    def get_issue(self, issue):
        return self.wiki.run(issue)

    def get_issue_description(self, category):
        return self.ddg.run(category)


functions_map = {
    "get_issue": IssueSearchClient.get_issue,
    "get_issue_description": IssueSearchClient.get_issue_description,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_issue",
            "description": "카테고리를 받으면 카테고리에 해당하는 최근 이슈를 찾아줍니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "카테고리를 받으면 카테고리에 해당하는 최근 이슈를 찾아줍니다."
                    }
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_issue_description",
            "description": "이슈를 받으면 이슈에 대한 설명을 해줍니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue": {
                        "type": "string",
                        "description": "이슈를 받습니다.",
                    }
                },
                "required": ["issue"],
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
    
    저는 2024년 4월 12일에 GPT4로 만들어진 이슈왕입니다. :dizzy:
    
    분야를 입력하시면 이슈를 빠르게 정리해드릴게요. :sparkles:
    
    저에게 물어보세요! :tulip: :tulip: :tulip:
    
    """
)

api_key = st.sidebar.text_input("당신의 OpenAI API Key를 입력하세요")

if api_key and api_key.startswith("sk-"):
    st.session_state["api_key"] = api_key
    client = OpenAI(api_key=api_key)

    assistant_id = "asst_KlzM0l2N7TIWoiIclGcutRQB"

    category = st.text_input("분야를 입력하세요.")

    if category:
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": f"나는 {category} 분야에 대해 알고 싶어요.",
                }
            ]
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )

        assistant = ThreadClient(client)
        run = assistant.wait_on_run(run, thread)

        if (run.status == "completed"):
            message = assistant.get_messages(thread.id)
            print(message)
