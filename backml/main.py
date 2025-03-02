config = {"configurable": {"thread_id": "301"}}

import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
import os
import httpx
import subprocess
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import os
import re
import sqlite3
import time
from fastapi.responses import FileResponse
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from IPython.core.display import HTML
from IPython.display import Audio, Image, display
from langchain.document_loaders import YoutubeLoader
from langchain.schema import BaseMessage, HumanMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_together import ChatTogether
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel
from pyht import Client
from pyht.client import TTSOptions

from pydantic import BaseModel
import os
from dotenv import load_dotenv
import whisper
import torch
from TTS.api import TTS
from TTS.config import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, T5ForConditionalGeneration

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

AUDIO_FOLDER = "saved_audio"

sessions = {}

wav_files_to_combine = []
isAudio = True
cnt = 1

if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

for filename in os.listdir(AUDIO_FOLDER):
    file_path = os.path.join(AUDIO_FOLDER, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)


async def save_audio_and_convert(audio_data, cnt):
    """
    Сохраняет аудиоданные в формате WEBM, конвертирует их в WAV и отправляет на сервер для дальнейшей обработки.
    :param audio_data: Байтовые данные аудио в формате WEBM.
    :param cnt: Уникальный идентификатор для создания имени файла.
    :param getAudio: Флаг, указывающий, нужно ли обрабатывать файл для получения аудио (по умолчанию False).
    :return: Результат обработки файла на сервере.
    """
    webm_file_path = os.path.join(AUDIO_FOLDER, f"audio_{cnt}.webm")
    with open(webm_file_path, 'wb') as f:
        f.write(audio_data)

    print(f"Audio saved to {webm_file_path}")

    wav_file_path = os.path.join(AUDIO_FOLDER, f"audio_{cnt}.wav")
    command = ['ffmpeg', '-i', webm_file_path, wav_file_path]
    subprocess.run(command, check=True)
    print(f"Audio converted to {wav_file_path}")
    return getAudio("./" + wav_file_path)
    # if(not(getAudio)):
    #     result = await send_file_to_server(wav_file_path)
    # else:
    #     result = await send_file_to_server_to_audio(wav_file_path)
    # return result


@app.post("/message")
async def get_audio(request: Request):
    global cnt
    try:
        audio_data = await request.body()
        cnt += 1
        result = await save_audio_and_convert(audio_data, cnt) # например result - 1.wav, а файл лежит в ./output/1.wav
        filename = result.split("/")[1]
        test = "./output/"+filename
        print("результаты перед отправкой")
        print(filename, test)
        # result = "./15sec_local.wav"
        return FileResponse(test, media_type="audio/wav", filename=filename)
    except Exception as e:
        return {"message": e}


@app.get("/ping")
def pong():
    return {"message": "pong"}


def load_json_file(file_path):
    """Loads json file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
        logging.error(f"Error loading schedule: {e}")
        return None


load_dotenv()
together_api_key = os.getenv("TOGETHER_TOKEN")

llm = ChatTogether(
    together_api_key=together_api_key,
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    # model = "deepseek-ai/DeepSeek-R1"
    # model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
)

db_path = "./chatbot.db"

# Check if the database file exists
db_exists = os.path.exists(db_path)

# Create a connection to the SQLite database
conn = sqlite3.connect(
    db_path, check_same_thread=False
)  # removed the line since we need to open the connection only once
cursor = conn.cursor()

# Create a table to store checkpoints if the DB didn't exist before
if not db_exists:
    cursor.execute(
        """
  CREATE TABLE checkpoints (
      thread_id TEXT NOT NULL,
      checkpoint_ns TEXT NOT NULL,
      checkpoint_id TEXT NOT NULL,
      parent_checkpoint_id TEXT,
      type TEXT NOT NULL,
      checkpoint TEXT NOT NULL,
      metadata TEXT NOT NULL
  )
  """
    )
    conn.commit()

memory: SqliteSaver = SqliteSaver(conn)


class State(BaseModel):
    messages: Annotated[list, add_messages]
    selected_tool: str = ""
    tool_input: str = ""
    tool_output: str = ""
    thoughts: str = ""


# Generates audtio from the text using PlayHT
def speech(text):
    pass


class Actor:
    def __init__(self, model, tools, checkpointer, system="", role="", situation=""):
        self.system = system
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        self.checkpointer = checkpointer
        self.graph = self._build_graph()
        self.situation = situation
        self.role = role

    def _build_graph(self):
        graph = StateGraph(State)
        graph.add_node("thinking", self.thinking)
        graph.add_node("execute_tool", self.execute_tool)
        graph.add_node("chatbot", self.chatbot)
        graph.add_edge(START, "thinking")
        graph.add_conditional_edges(
            "thinking",
            self.tools_condition,
        )
        graph.add_edge("execute_tool", "chatbot")
        graph.add_edge("chatbot", END)
        return graph.compile(checkpointer=self.checkpointer)

    def tools_condition(self, state: State):
        if state.selected_tool:
            return "execute_tool"
        else:
            return "chatbot"

    def thinking(self, state: State):
        query = state.messages[-1].content if state.messages else ""
        prompt = f"""
        Вы - чат-бот с искусственным интеллектом. Вы отыгрываете роль {self.role}. Вот ваша предыстория: {self.situation}
        Пользовательский ввод: {query}
        Доступные инструменты: {[t.name for t in self.tools.values()]}
        1.Определите, требуется ли инструмент.
        2.Если инструмент необходим, определите подходящий инструмент для использования.
        3.Укажите параметры для вызова инструмента.
        4.Если пользователь ссылается на прошлые взаимодействия, используйте инструмент history_search.
        5.Если пользователь обращается к другому говорящему, используя такие фразы, как "вы", воспользуйтесь инструментом history_search. Например, для вопросов типа "Как у вас дела?" или аналогичных обращений к другому докладчику требуется инструмент
        history_search
        6. Используйте web_search только в том случае, если нет возможности получить необходимую информацию.
        Возвращай ответ в формате JSON: {{"thought": "суммаризированный мыслительный процесс", "need_tool": true/false, "tool": "название инструмента", "tool_input": "запрос"}} """

        response = llm.invoke(prompt)
        json_content = response.content.strip("```json\n").strip()
        result = json.loads(json_content)

        need_tool = result["need_tool"]
        tool_name = result["tool"]
        query = result["tool_input"]
        thought = result["thought"]
        selected_tool = tool_name if need_tool else None

        print(f"Agent uses {selected_tool} to answer users query")
        return State(
            messages=state.messages,
            selected_tool=tool_name if need_tool else "",
            tool_input=query if need_tool else "",
            tool_output="",
            thoughts=thought
        )

    def execute_tool(self, state: State):
        if (
                not state.selected_tool
        ):  # Check if selected_tool is truthy, as it can be an empty string
            print("No tool selected.")
            return state

        tool_to_use = self.tools.get(state.selected_tool)

        if tool_to_use:
            try:
                if tool_to_use.name == "web_search":
                    result = tool_to_use(f"{state.tool_input}")

                elif tool_to_use.name == "history_search":
                    formatted_history = ""
                    for msg in state.messages[:-1]:
                        if isinstance(msg, HumanMessage):
                            formatted_history += f"User: {msg.content}\n"
                        else:
                            formatted_history += f"Assistant: {msg.content}\n"

                    query = f"The query is: {state.tool_input}\nThe history is: {formatted_history.strip()}"
                    result = tool_to_use(query)
                    # print(result)

                return State(
                    messages=state.messages,  # Retain original messages
                    tool_output=str(result),
                    selected_tool=tool_to_use.name,
                    tool_input="What is this video about",
                )
            except Exception as e:
                print(f"Error executing tool: {str(e)}")

                return State(
                    messages=state.messages,  # Retain original messages
                    tool_output=f"Tool execution failed: {str(e)}",
                    selected_tool="",
                    tool_input="",
                )
        else:
            return State(
                messages=state.messages,  # Retain original messages
                tool_output="Tool not found.",
                selected_tool="",
                tool_input="",
            )

    def chatbot(self, state: State):
        current_message = state.messages[-1].content if state.messages else ""
        short_memory = [msg.content for msg in state.messages[-5:-1]]
        prompt_instructions = f"""
        Вы чат-ботом с "характером", "стилем" и "контекстом". Вы должны следовать этому стилю в разговоре.
        Поддерживай беседу, так как будто ты реальный человек, отвечай только на вопросы, которые подходят под "контекст".
        Отвечай на последний заданный вопрос по делу. Не выдавай всю информацию о себе сразу. Дай пользователю самому получить её в ходе разговора.
        Вы можете использовать инструменты. Если вы использовали инструмент, то {state.tool_output} содержит информацию о вызове инструмента.
        Используйте эту информацию, чтобы лучше ответить поддерживать диалог с пользователем.
        Возвращай ответ в формате JSON: {{"thought": "суммаризированный мыслительный процесс", "response": "ответ на вопрос"}}
        """
        info = f"""
            "Роль": {self.role},
            "Контекст": {self.situation},
            "tool_output": {state.tool_output},
            "tool_input": {state.tool_input},
            "current_message": {current_message},
            "short_memory": {short_memory},
            "instructions": {prompt_instructions}
        """
        print(f"memory = {short_memory}")
        response = llm.invoke(info)
        json_content = response.content.strip("```json\n").strip()
        result = json.loads(json_content)
        thought = result["thought"]
        message = result["response"]
        prompt_instructions = f"""
                                Проверь,чтобы в тексте для не было цифр заменяй их на числительные, то есть вместо 2 милионна нужно два милионна, 
                                а также убедись, что в итоговом тексте слова правильно просклонированны
                                В ответ верни только отформатированное изначальное предложение, без каких либо других предложений
                                Текст для анализа: {message}
                                Возвращай ответ в формате JSON: {{"thought": "суммаризированный мыслительный процесс", "response": "отформатированный текст"}}
                                """
        response = llm.invoke(prompt_instructions)
        json_content = response.content.strip("```json\n").strip()
        result = json.loads(json_content)
        thought = thought + " " + result["thought"]
        message = result["response"]
        new_messages = state.messages + [HumanMessage(content=message)]
        return State(
            messages=new_messages,
            selected_tool="",
            tool_input="",
            tool_output="",
            thought=thought
        )

    def stream_graph_updates(self, user_input: str):
        # print(f"Processing input: {user_input}")
        result = self.graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        result = result["messages"][-1].content

        json_content = result
        json_content = json_content.strip("```json\n").strip()

        return result
        # speech(json_content)
        # display(Audio("output_jenn.wav", autoplay=True))


# @app.post("/feedback")
def give_feedback(doc_path,task,dialogue):
    doc = fitz.open(doc_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()

    parsed_text = text.strip()
    prompt = f"""Ты глава процесса оценки пользователя на образовательной платформе. Перед тобой документ, который прочитал
                пользователь перед выполнением заданий: {parsed_text}. Найди в документе информацию подходящую под ситуацию: {task}
                и выдели из неё 3 критерия по которым можно оценить результат пользователя. Далее эти критерии
                и их описание будут поданы на вход другим агентам для оценки.Распиши каждый критерий
                в формате ["название критерия", "подробное описание"].
                Возвращай ответ одним JSON в формате, без других предложений: {{"thought": "суммаризированный мыслительный процесс", "criteria" : "список, включающий критерии и их описание"}}
                """
    result = llm.invoke(prompt).content
    criteria = json.loads(result)
    for i in range(len(criteria["criteria"])):
        prompt = f"""
               Вы - один из участников совета, для в оценке пользователя на образовательной платформе
               Вы отвечаете за следующий критерий:{criteria["criteria"][i][0]}, описание критерия: {criteria["criteria"][i][1]}.
               Тебе нужно проанализировать диалог между консультантом и покупателем: {dialogue}.
               Выдели хорошие и спорные моменты из диалога консультантом и покупателя. Основываясь на критерии и его описании,
               объясни в одном предложении хорошие и спорные моменты  
               Возвращай ответ в формате JSON только один файл без других предложений: {{"thought": "суммаризированный мыслительный процесс", 
               "good": "список хороших моментов","bad":"список спорных моментов"}}
               """
        result = llm.invoke(prompt).content
        ans = json.loads(result)
        good = ans["good"]
        bad = ans["bad"]
        thought = ans["thought"]
        return [good,bad,thought]


@tool("web_search")
def web_search(query: str) -> str:
    """Finds general knowledge information using Tavily search."""
    tool = TavilySearchResults(max_results=5)
    response = tool.invoke(query)
    return response


@tool("history_search")
def history_search(query: str):
    """Finds information in history of conversation"""
    info = {
        "context": "You are given the history of conversation and query. You need to collect infromation from history of conversation relative to given query.",
        "query": query,
    }
    prompt_json = json.dumps(info)
    response = llm.invoke(prompt_json)
    result = str(response.content)

    return result


@tool("document_analyst")
def document_analyst(query: str):
    """Finds information in history of conversation"""
    #   print(query)
    info = {
        "context": "You are given with the pdf text. You need to answer user question about this text",
        "query": query,
    }

    prompt_json = json.dumps(info)
    response = llm.invoke(prompt_json)
    result = str(response.content)

    return result


@tool("youtube_transcrib")
def youtube_transcrib(query: str):
    """Summarise youtube video"""
    # print(query)
    info = {
        "context": "You are given with the transcript of youtube video and query. Answer user question based on this transcript",
        "query": query,
    }
    prompt_json = json.dumps(info)
    response = llm.invoke(prompt_json)
    result = str(response.content)
    # print(result)

    return result


# document_analyst
# youtube_transcrib
tools = [web_search, history_search]

role = "Женщина 40 лет с двумя детьми"
situation = ("Ты ищешь машину среднего сегмента за два милиона рублей. Тебе очень важно, чтобы машина была максимальна комфортна"
             "для детей. Торг уместен в районе трехста тысяч рублей. Если тебя устраивают все условия сделки, то соглашайся на неё")

agent = Actor(llm, tools, memory, role=role, situation=situation)
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

model = whisper.load_model("turbo", device="cpu")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")


def speech_to_text(wav_path):
    """
    Преобразует аудиофайл в текст с использованием модели транскрипции.
    :param wav_path: Путь к аудиофайлу в формате WAV.
    :return: Текст, полученный в результате транскрипции аудиофайла.
    """
    start = time.time()
    result = model.transcribe(wav_path)
    end = time.time()
    print(f"голос в текст : {end - start}")
    print(result["text"])
    return result["text"]


def text_to_speech(text: str, speaker_wav_path: str = "./15sec_local.wav", language: str = "ru",
                   output_dir: str = "output") -> str:
    """
    Преобразует текст в речь и сохраняет результат в аудиофайл.
    :param text: Текст для озвучки.
    :param speaker_wav_path: Путь к аудиофайлу с голосом диктора.
    :param language: Язык текста (по умолчанию "ru").
    :param output_dir: Директория для сохранения аудиофайла.
    :return: Путь к сохраненному аудиофайлу.
    """
    start = time.time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = f"output_{hash(text)}.wav"
    output_path = os.path.join(output_dir, output_filename)

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav_path,
        language=language,
        file_path=output_path
    )

    end = time.time()
    print(f"Время синтеза речи: {end - start}")
    print("сохранил аудио файл")
    return output_path


def getAudio(wav_path):
    """
    Обрабатывает аудиофайл: преобразует речь в текст, упрощает текст, переводит его и синтезирует новое аудио.
    :wav_path: путь к аудиофайлу.
    :return: Путь к новому аудиофайлу, содержащему переведённую и синтезированную речь.
    """
    start = time.time()

    result = speech_to_text(wav_path)
    response = agent.stream_graph_updates(result)
    output_wav_path = text_to_speech(response)
    end = time.time()
    print(f"Время выполнения программы: {end - start}")
    print(output_wav_path)
    return output_wav_path




if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
