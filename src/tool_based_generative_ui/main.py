#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv(override=True)
from crewai.flow import Flow, start
from crewai import LLM
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = LLM(
    model="gpt-4o"
)

class ToolBasedGenerativeUIFlow(Flow):

    @start()
    def chat(self):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ]

        logger.info(f"Initial input state: {self.state}")
        logger.info(f"Raw state dump: {json.dumps(self.state, indent=2)}")
        logger.info(f"Messages in state: {self.state.get('messages', [])}")

        logger.info(f"####STATE####\n{json.dumps(self.state, indent=2)}")

        if self.state.get('messages'):
            messages.append(self.state['messages'][-1])

        response = llm.call(messages)

        if 'messages' not in self.state:
            self.state['messages'] = []
        self.state['messages'].append({
            "role": "assistant",
            "content": response
        })

        self.state['messages'].append({
            "role": "assistant",
            "content": response
        })

        return response


def kickoff():
    tool_based_generative_ui_flow = ToolBasedGenerativeUIFlow()
    response = tool_based_generative_ui_flow.kickoff({
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
    })
    logger.info(f"Response: {response}")


def plot():
    tool_based_generative_ui_flow = ToolBasedGenerativeUIFlow()
    tool_based_generative_ui_flow.plot()


if __name__ == "__main__":
    kickoff()
