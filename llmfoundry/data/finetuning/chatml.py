# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import List


class ChatFormatter:
    """A class for formatting the chat history.

    Args:
        system: The system prompt. If None, a default ChatML-formatted prompt is used.
        user: The user prompt. If None, a default ChatML value is used.
        assistant: The assistant prompt. If None, a default ChatML value is used.

    Attributes:
        system: The system prompt.
        user: The user prompt.
        assistant: The assistant prompt.
        response_prefix: The response prefix (anything before {} in the assistant format string)
    """

    def __init__(self, system: str, user: str, assistant: str) -> None:
        self.system = system if system else '<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>\n'
        self.user = user if user else '<|im_start|>user\n{}<|im_end|>\n'
        self.assistant = assistant if assistant else '<|im_start|>assistant\n{}<|im_end|>\n'
        self.response_prefix = self.assistant.split('{}')[0]

    def as_string(self, history: List[List[str]]) -> str:
        """Returns the chat history as a string.
        
        Args:
            history: A list of lists of strings. Each inner list contains two strings: the user input and the assistant response.
            
        Returns:
            The chat history as a string, formatted in ChatML syntax.
        """
        text = self.system + ''.join([
            '\n'.join([
                self.user.format(item[0]),
                self.assistant.format(item[1]),
            ]) for item in history[:-1]
        ])
        text += self.user.format(history[-1][0])
        text += self.response_prefix
        return text


class ChatMLFormatter(ChatFormatter):
    """A class for formatting the chat history in ChatML syntax.

    Args:
        system: The system prompt. If None, a default ChatML-formatted prompt is used.
        user: The user prompt. If None, a default ChatML value is used.
        assistant: The assistant prompt. If None, a default ChatML value is used.
    """

    system_fmt = '<|im_start|>system\n{}<|im_end|>\n'

    def __init__(self, system: str = None, user: str = None, assistant: str = None) -> None:
        if system:
            system = self.system_fmt.format(system)
        super().__init__(system, user, assistant)
        self.response_suffix = self.assistant.split('{}')[1]

    def format_response(self, response: str):
        """Formats a response in ChatML syntax.

        It assumes that the prompt contained the prefix of the response.
        
        Args:
            response: The response to format.
            
        Returns:
            The response, formatted in ChatML syntax.
        """
        return response + self.response_suffix