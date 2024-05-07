import csv
import os
import re

from autogen import UserProxyAgent, AssistantAgent
from dotenv import load_dotenv


class Reflection:
    def __init__(self, prompter_system_prompt, critic_system_prompt, results_folder):
        load_dotenv(override=True)  # notice the override=True
        self.prompter_system_prompt = prompter_system_prompt
        self.critic_system_prompt = critic_system_prompt
        self.results_folder = results_folder
        self.category_description = dict()
        self.user_proxy_agent = None
        self.prompter_agent = None
        self.critic_agent = None

    def create_user_proxy_agent(self):
        self.user_proxy_agent = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
            code_execution_config=False,
        )

    def create_prompter_agent(self):
        self.prompter_agent = AssistantAgent(
            name="Prompter",
            llm_config={"config_list": [{"model": "gpt-4-turbo", "api_key": os.environ.get("OPENAI_API_KEY")}]},
            system_message=self.prompter_system_prompt,
        )

    def create_critic_agent(self):
        self.critic_agent = AssistantAgent(
            name="Critic",
            llm_config={"config_list": [{"model": "gpt-4-turbo", "api_key": os.environ.get("OPENAI_API_KEY")}]},
            system_message=self.critic_system_prompt,
        )

    def reflection_message(self, recipient, messages, sender, config):
        print("Reflecting...")
        reflection_message = {}
        # message = (recipient.chat_messages_for_summary(sender)[-2]['content'] + "prompt: "
        #            + recipient.chat_messages_for_summary(sender)[-1]['content'])

        messages.append({"role": "user", "content": "Reflect and provide critique on the analysis"})
        return messages

    def all_messages(self, recipient, sender, summary_dict):
        messages = sender.chat_messages_for_summary(recipient)
        return messages

    def register_nested_agent(self, agent):
        # "summary_method" is only called at the end?
        agent.register_nested_chats(
            [{"recipient": self.critic_agent, "message": self.reflection_message,
              "max_turns": 1}],
            trigger=self.prompter_agent,
        )

    def run(self):
        self.create_user_proxy_agent()
        self.create_prompter_agent()
        self.create_critic_agent()
        self.register_nested_agent(self.user_proxy_agent)

        # Get the directory of the current script, must have for portability
        dir_path = os.path.dirname(os.path.realpath(__file__))

        # res = self.user_proxy_agent.initiate_chat(recipient=self.prompter_agent, message=message, max_turns=2,
        #                                           summary_method="last_msg")
        #
        # content = res.chat_history[0].get("content")
        # lines = content.splitlines()
        # non_empty_lines = [line.strip() for line in lines if line.strip()]
        # # Rule 1: Replace multiple \n with a single one.
        # # content = re.sub(r'\n+', '\n', content)
        # # Rule 2: Replace multiple spaces or tabs (in the middle of lines) with a single space.
        # # We avoid replacing single newlines by using a negative lookahead and lookbehind assertion for \n.
        # # Rule 3: Remove spaces and tabs from the beginning and end of the string.
        #
        # new_file_path = os.path.join(prompts_folder, f"prompt_{counter}.txt")
        # with open(new_file_path, 'w', encoding='utf-8') as f:
        #     # print("*" * 45, "Final Result", "*" * 45)
        #     for line in non_empty_lines:
        #         line = line.strip()
        #         line = re.sub(r'(?<!\n)[ \t]+(?!\\n)', ' ', line)
        #         f.write(f"{line:>{4 + len(line)}}\n")
        #     prompt = "prompt: " + res.chat_history[-1].get('content').strip()
        #     formatted_prompt = f"{prompt:>{4 + len(prompt)}}"
        #     # print(formatted_prompt)
        #     f.write(f"{formatted_prompt}\n")
        #     # print("*" * 100)
