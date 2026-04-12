"""
실시간 뉴스 분석 & 이메일 자동화 에이전트
Gradio 웹 UI — Railway 배포용
"""

import os
import gradio as gr
from main import run_pipeline

def analyze_news(keyword, target_reader, send_email, send_to,
                 openai_key, tavily_key, gmail_address, gmail_pw):
    if not keyword.strip():
        return "❌ 키워드를 입력해주세요.", ""
    if not target_reader.strip():
        return "❌ 독자 페르소나를 입력해주세요.", ""

    if openai_key and openai_key.strip():
        os.environ['OPENAI_API_KEY'] = openai_key.strip()
    if tavily_key and tavily_key.strip():
        os.environ['TAVILY_API_KEY'] = tavily_key.strip()
    if gmail_address and gmail_address.strip():
        os.environ['GMAIL_ADDRESS'] = gmail_address.strip()
    if gmail_pw and gmail_pw.strip():
        os.environ['GMAIL_APP_PASSWORD'] = gmail_pw.strip()

    if not os.environ.get('OPENAI_API_KEY'):
        return "❌ OpenAI API 키가 없어요. 아래 API 키 설정에서 입력해주세요.", ""
    if not os.environ.get('TAVILY_API_KEY'):
        return "❌ Tavily API 키가 없어요. 아래 API 키 설정에서 입력해주세요.", ""

    email_to = None
    if send_email and send_to and send_to.strip():
        email_to = send_to.strip()

    try:
        result = run_pipeline(
            keyword=keyword.strip(),
            target_reader=target_reader.strip(),
            send_to_email=email_to
        )

        subject_output = f"📧 제목: {result['subject']}"
        body_output = result['body']

        if email_to:
            if result['send_success']:
                body_output += f"\n\n---\n📬 이메일 발송 성공 → {email_to}"
            else:
                body_output += f"\n\n---\n❌ 이메일 발송 실패. Gmail 앱 비밀번호를 확인해주세요."

        return subject_output, body_output

    except Exception as e:
        return f"❌ 오류 발생: {str(e)}", ""


with gr.Blocks(title="AI 뉴스 인사이트 에이전트") as demo:

    gr.Markdown("""
    # 📰 AI 뉴스 인사이트 에이전트
    키워드를 입력하면 최신 뉴스를 수집·분석해서 뉴스레터 이메일을 자동 생성합니다.

    **기술 스택**: CrewAI · GPT-4o-mini · Tavily RAG · Prompt Chaining
    """)

    with gr.Row():
        keyword_input = gr.Textbox(
            label="🔍 분석할 키워드",
            placeholder="예: 삼성전자 반도체",
            scale=2
        )
        reader_input = gr.Textbox(
            label="👤 독자 페르소나",
            placeholder="예: IT 업계 투자자",
            scale=2
        )

    with gr.Row():
        send_toggle   = gr.Checkbox(label="📧 이메일 발송", value=False)
        send_to_input = gr.Textbox(label="수신 이메일", placeholder="받을 이메일 주소", scale=3)

    run_btn = gr.Button("🚀 뉴스 분석 시작", variant="primary", size="lg")

    gr.Markdown("---")

    subject_output = gr.Textbox(label="📧 이메일 제목", interactive=False)
    body_output    = gr.Textbox(label="📝 이메일 본문", lines=20, interactive=False)

    with gr.Accordion("🔑 API 키 설정 (환경변수 미설정 시)", open=False):
        gr.Markdown("Railway Variables에 등록했으면 여기 입력 안 해도 돼요!")
        with gr.Row():
            openai_input  = gr.Textbox(label="OpenAI API Key",   type="password", placeholder="sk-...")
            tavily_input  = gr.Textbox(label="Tavily API Key",   type="password", placeholder="tvly-...")
        with gr.Row():
            gmail_input   = gr.Textbox(label="Gmail 주소",        placeholder="example@gmail.com")
            gmailpw_input = gr.Textbox(label="Gmail 앱 비밀번호", type="password", placeholder="16자리")

    run_btn.click(
        fn=analyze_news,
        inputs=[
            keyword_input, reader_input,
            send_toggle, send_to_input,
            openai_input, tavily_input, gmail_input, gmailpw_input
        ],
        outputs=[subject_output, body_output]
    )

    gr.Markdown("""
    ---
    ### 🏗️ 아키텍처
    ```
    키워드 입력
        ↓
    Agent A (Researcher) — Tavily로 실시간 뉴스 수집 + 정제 (RAG)
        ↓
    Agent B (Writer) — 뉴스레터 이메일 초안 생성
        ↓
    Gmail 자동 발송
    ```
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
