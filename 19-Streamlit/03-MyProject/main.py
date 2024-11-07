import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote.prompts import load_prompt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import glob

# API KEY 정보 로드
load_dotenv()

# 제목
st.title("나만의 AI 어시스턴트")

# 처음 한번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# 사이드바
with st.sidebar:
    clear_btn = st.button("대화 초기화")

    prompt_files = glob.glob("prompts/*.yaml")
    selected_prompt = st.selectbox("프롬프트를 선택해 주세요", prompt_files, index=0)
    task_input = st.text_input("TASK 입력")


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def create_chain(prompt_filepath, task=""):
    prompt = load_prompt(prompt_filepath, encoding="utf-8")

    if task:
        prompt = prompt.partial(task=task)

    # GPT
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 출력 파서
    output_parser = StrOutputParser()

    # 체인 생성
    chain = prompt | llm | output_parser
    return chain


# 초기화 버튼 누를 시
if clear_btn:
    st.session_state["messages"] = []

print_messages()

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요")

# 사용자의 입력이 있으면
if user_input:
    # 사용자 입력
    st.chat_message("user").write(user_input)

    # 체인 생성
    chain = create_chain(selected_prompt, task=task_input)

    # 스트리밍 호출
    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 토큰을 스트리밍 출력한다.
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)
    # 대화기록 저장
    add_message("user", user_input)
    add_message("assistant", ai_answer)
