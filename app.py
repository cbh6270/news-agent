"""
실시간 뉴스 분석 & 이메일 자동화 에이전트
Streamlit 웹 UI
"""

import os
import streamlit as st
from main import run_pipeline

# ── 페이지 기본 설정 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="AI 뉴스 인사이트 에이전트",
    page_icon="📰",
    layout="centered"
)

# ── 헤더 ─────────────────────────────────────────────────────────────
st.title("📰 AI 뉴스 인사이트 에이전트")
st.caption("키워드를 입력하면 최신 뉴스를 수집·분석해서 뉴스레터 이메일을 자동 생성합니다.")
st.divider()

# ── API 키 설정 ───────────────────────────────────────────────────────
# 로컬 실행 시: 직접 입력
# Streamlit Cloud 배포 시: Secrets에서 자동으로 읽어옴
with st.sidebar:
    st.header("🔑 API 키 설정")
    st.caption("로컬 실행 시 여기에 입력하세요. Streamlit Cloud 배포 시 자동 적용됩니다.")

    openai_key  = st.text_input("OpenAI API Key",  type="password", placeholder="sk-...")
    tavily_key  = st.text_input("Tavily API Key",  type="password", placeholder="tvly-...")
    gmail_addr  = st.text_input("Gmail 주소",       placeholder="example@gmail.com")
    gmail_pw    = st.text_input("Gmail 앱 비밀번호", type="password", placeholder="16자리")

    if st.button("키 저장", use_container_width=True):
        os.environ['OPENAI_API_KEY']    = openai_key
        os.environ['TAVILY_API_KEY']    = tavily_key
        os.environ['GMAIL_ADDRESS']     = gmail_addr
        os.environ['GMAIL_APP_PASSWORD']= gmail_pw
        st.success("✅ API 키 저장 완료!")

    st.divider()
    st.caption("💡 Streamlit Cloud 배포 후에는 Secrets에 등록하면 사이드바 입력 불필요")

# ── 메인 입력 폼 ──────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    keyword = st.text_input(
        "🔍 분석할 키워드",
        placeholder="예: 삼성전자 반도체",
        help="뉴스를 검색할 주제를 입력하세요"
    )

with col2:
    target_reader = st.text_input(
        "👤 독자 페르소나",
        placeholder="예: IT 업계 투자자",
        help="뉴스레터를 받을 독자를 입력하세요"
    )

send_email = st.toggle("📧 이메일 발송", value=True, help="완성된 뉴스레터를 Gmail로 발송합니다")

if send_email:
    send_to = st.text_input(
        "수신 이메일 주소",
        placeholder="받을 이메일 주소 입력",
        value=os.environ.get('GMAIL_ADDRESS', '')
    )
else:
    send_to = None

st.divider()

# ── 실행 버튼 ─────────────────────────────────────────────────────────
run_button = st.button(
    "🚀 뉴스 분석 시작",
    type="primary",
    use_container_width=True,
    disabled=not keyword or not target_reader
)

# ── 실행 ─────────────────────────────────────────────────────────────
if run_button:

    # API 키 환경변수 설정 (Streamlit Cloud Secrets 자동 적용)
    if not os.environ.get('OPENAI_API_KEY'):
        try:
            os.environ['OPENAI_API_KEY']     = st.secrets['OPENAI_API_KEY']
            os.environ['TAVILY_API_KEY']     = st.secrets['TAVILY_API_KEY']
            os.environ['GMAIL_ADDRESS']      = st.secrets['GMAIL_ADDRESS']
            os.environ['GMAIL_APP_PASSWORD'] = st.secrets['GMAIL_APP_PASSWORD']
        except:
            st.error("❌ API 키가 설정되지 않았어요. 사이드바에서 키를 입력해주세요.")
            st.stop()

    # 실행 상태 표시
    st.info(f"⚙️ **{keyword}** 뉴스 분석 중... (1~3분 소요)")

    # 진행 상황 표시
    progress = st.empty()
    with progress.container():
        st.write("🔍 **Agent A**: Tavily로 최신 뉴스 수집 중...")
        agent_a_status = st.empty()
        st.write("✍️ **Agent B**: 뉴스레터 초안 작성 중...")
        agent_b_status = st.empty()

    # 파이프라인 실행
    with st.spinner("AI 에이전트 실행 중..."):
        try:
            result = run_pipeline(
                keyword=keyword,
                target_reader=target_reader,
                send_to_email=send_to
            )

            # 진행 상황 지우고 결과 표시
            progress.empty()

            st.success("✅ 분석 완료!")
            st.divider()

            # 이메일 제목
            st.subheader("📧 생성된 이메일")
            st.markdown(f"**제목:** {result['subject']}")
            st.divider()

            # 이메일 본문
            st.markdown(result['body'])
            st.divider()

            # 발송 결과
            if send_to:
                if result['send_success']:
                    st.success(f"📬 이메일 발송 성공 → {send_to}")
                else:
                    st.error("❌ 이메일 발송 실패. Gmail 앱 비밀번호를 확인해주세요.")

            # 복사용 텍스트
            with st.expander("📋 전체 텍스트 복사하기"):
                st.code(f"제목: {result['subject']}\n\n{result['body']}", language=None)

        except Exception as e:
            progress.empty()
            st.error(f"❌ 오류 발생: {e}")
            st.caption("API 키가 올바른지, 인터넷 연결이 됐는지 확인해주세요.")

# ── 하단 설명 ─────────────────────────────────────────────────────────
st.divider()
with st.expander("🏗️ 기술 스택 보기"):
    st.markdown("""
    | 구성 요소 | 기술 |
    |---|---|
    | LLM | GPT-4o-mini (OpenAI) |
    | 멀티 에이전트 | CrewAI |
    | 실시간 뉴스 검색 | Tavily API (RAG) |
    | 이메일 발송 | Gmail SMTP |
    | 웹 UI | Streamlit |

    **Agent A (Researcher)**: Tavily로 뉴스 수집 → 정제 → 핵심 수치 추출

    **Agent B (Writer)**: 인사이트 → 3줄 요약 → 뉴스레터 이메일 작성
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
