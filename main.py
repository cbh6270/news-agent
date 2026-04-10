"""
실시간 뉴스 분석 & 이메일 자동화 에이전트
GitHub Actions에서 매일 자동 실행되는 메인 스크립트
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import TavilySearchTool

# ── API 키 설정 ────────────────────────────────────────────────────────
# GitHub Actions에서는 Colab Secrets 대신 GitHub Secrets에서 읽어옴
# GitHub 레포 → Settings → Secrets and variables → Actions 에서 등록
OPENAI_API_KEY  = os.environ.get('OPENAI_API_KEY')
TAVILY_API_KEY  = os.environ.get('TAVILY_API_KEY')
GMAIL_ADDRESS   = os.environ.get('GMAIL_ADDRESS')
GMAIL_APP_PW    = os.environ.get('GMAIL_APP_PASSWORD')

# 키 로드 확인
print(f'✅ OPENAI_API_KEY     로드 완료 (길이: {len(OPENAI_API_KEY)}자)')
print(f'✅ TAVILY_API_KEY     로드 완료 (길이: {len(TAVILY_API_KEY)}자)')
print(f'✅ GMAIL_ADDRESS      로드 완료 ({GMAIL_ADDRESS[:4]}...)')
print(f'✅ GMAIL_APP_PASSWORD 로드 완료 (길이: {len(GMAIL_APP_PW)}자)')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

# ── LLM 초기화 ────────────────────────────────────────────────────────
# ✅ 모델 변경이 필요하면 여기만 수정
MODEL_NAME = 'gpt-4o-mini'

llm_researcher = ChatOpenAI(model=MODEL_NAME, temperature=0.2)
llm_writer     = ChatOpenAI(model=MODEL_NAME, temperature=0.5)

# ── CrewAI 에이전트 & 도구 정의 ───────────────────────────────────────
tavily_tool = TavilySearchTool(
    max_results=5,
    search_depth='advanced',
    include_raw_content=True
)

researcher = Agent(
    role='시니어 뉴스 리서처',
    goal=(
        '주어진 키워드로 최신 뉴스를 수집하고, '
        '광고·노이즈·중복을 제거한 뒤 '
        '핵심 수치와 비즈니스 인사이트만 구조화된 형태로 전달한다.'
    ),
    backstory=(
        '10년 경력의 데이터 저널리스트 출신으로, '
        '수천 건의 기사에서 핵심을 뽑아내는 훈련을 받았다. '
        '할루시네이션을 극도로 경계하며, 수집한 사실만 보고한다.'
    ),
    tools=[tavily_tool],
    llm=MODEL_NAME,
    llm_kwargs={'temperature': 0.2},
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role='B2B 뉴스레터 에디터',
    goal=(
        '리서처가 전달한 인사이트를 바탕으로 '
        '비즈니스 의사결정자가 30초 안에 핵심을 파악할 수 있는 '
        '전문적이고 읽기 쉬운 뉴스레터 이메일 초안을 작성한다.'
    ),
    backstory=(
        'Forbes, Harvard Business Review 스타일의 글쓰기에 특화된 에디터로, '
        '복잡한 비즈니스 정보를 명확하고 설득력 있게 전달하는 능력을 갖췄다. '
        '독자의 시간을 존중하여 핵심부터 전달하는 글쓰기 원칙을 따른다.'
    ),
    tools=[],
    llm=MODEL_NAME,
    llm_kwargs={'temperature': 0.5},
    verbose=True,
    allow_delegation=False
)

print('✅ CrewAI 에이전트 정의 완료')


# ── Task 정의 ─────────────────────────────────────────────────────────
def create_tasks(keyword: str, target_reader: str) -> tuple:
    task_research = Task(
        description=f"""
        [주제 키워드]: {keyword}
        [목표 독자]:  {target_reader}

        아래 4단계를 순서대로 수행하라.

        ## 1단계 - Retrieval (검색)
        Tavily로 "{keyword} 최신 뉴스"를 검색하여 관련성 높은 기사 5개를 수집한다.
        반드시 실제 검색 결과를 사용하고, 학습 데이터에서 내용을 지어내지 않는다.

        ## 2단계 - 데이터 정제 (Prompt Chain 1)
        수집한 기사에서 다음 항목을 제거한다:
        - 광고성 문구, 구독 유도 문구
        - 중복되는 내용
        - 기자 이름, 발행 시각 등 메타정보
        - 의미 없는 수식어

        ## 3단계 - 핵심 수치 추출 (Prompt Chain 2)
        정제된 내용에서 의사결정에 필요한 정보만 뽑아 구조화한다:
        - 구체적인 수치 (%, 금액, 증감률, 날짜)
        - 핵심 사건 3가지
        - 시장 트렌드 방향성
        - 주목할 리스크

        ## 4단계 - 출처 기록
        사용한 기사의 URL을 모두 기록한다. (팩트체크 근거)

        ## 5단계 - 언어 통일
        위에서 추출한 모든 수치와 내용을 한국어로 번역하라.
        단, 고유명사(회사명, 인명, 브랜드명)와 URL은 영어 그대로 유지한다.
        금액은 "1억 달러", "170억 달러" 처럼 한국어 단위로 표기한다.
        """,
        expected_output=f"""
        아래 형식을 정확히 지켜서 출력하라:

        ## 📊 핵심 수치
        아래 형식으로 정렬해서 출력하라 (모두 한국어로):
        이모지 항목명 : 수치 (한 줄 설명)
        예시)
        💰 총 투자액    : 170억 달러 (텍사스 팹 기준)
        👥 고용 인원    : 1,500명 (2026년까지 예정)

        ## 📰 주요 사건/발표 TOP 3
        1. (가장 중요한 사건)
        2.
        3.

        ## 💡 시장 인사이트
        - (트렌드 및 방향성 2~3가지)

        ## ⚠️ 주목할 리스크
        - (부정적 신호 1~2가지)

        ## 🔗 참고 출처
        - (URL 목록)
        """,
        agent=researcher
    )

    task_write = Task(
        description=f"""
        [주제 키워드]: {keyword}
        [목표 독자]:  {target_reader}

        리서처(Agent A)가 전달한 인사이트를 바탕으로 아래 2단계를 수행하라.
        절대로 리서처가 제공하지 않은 정보를 추가하거나 지어내지 않는다.

        ## 1단계 - 3줄 핵심 요약 (Prompt Chain 3)
        {target_reader}가 30초 안에 핵심을 파악할 수 있는 3줄 요약 작성:
        - 각 줄은 하나의 메시지만 담을 것
        - 반드시 구체적인 수치 포함
        - 각 줄 앞에 관련 이모지 1개

        ## 2단계 - 이메일 초안 작성 (Prompt Chain 4)
        아래 구조로 뉴스레터 이메일 전체를 작성한다:
        - 클릭하고 싶은 이메일 제목 (숫자 또는 임팩트 있는 표현 포함)
        - 인사말 및 오늘의 주제 소개
        - 핵심 3줄 요약 섹션
        - 주요 수치 & 팩트 섹션
        - 에디터 인사이트 (독자에게 의미하는 바)
        - 주목할 리스크 섹션
        - 출처 링크
        - 마무리 인사
        """,
        expected_output=f"""
        아래 형식을 정확히 지켜서 출력하라:

        ===이메일 제목===
        (제목 내용)
        ===이메일 제목 끝===

        ===이메일 본문===
        안녕하세요, {target_reader}님 👋

        오늘의 주제: **{keyword}**

        📌 **오늘의 핵심 3줄**
        (3줄 요약)

        ---

        📊 **주요 수치 & 팩트**
        아래 형식으로 표처럼 정렬해서 출력하라 (반드시 한국어로):
        이모지 항목명 : 수치 (한 줄 설명)
        예시)
        💰 총 투자액    : 170억 달러 (텍사스 팹 기준)
        👥 고용 인원    : 1,500명 (2026년까지 예정)

        💡 **에디터 인사이트**
        (독자에게 의미하는 바 2~3문장)

        ⚠️ **주목할 리스크**
        (리스크 1~2가지)

        🔗 **참고 출처**
        (URL 목록)

        ---
        본 뉴스레터는 AI 에이전트가 자동 수집·분석한 내용입니다.
        ===이메일 본문 끝===
        """,
        agent=writer,
        context=[task_research]
    )

    return task_research, task_write


# ── Gmail 발송 모듈 ───────────────────────────────────────────────────
def parse_email_result(crew_output: str) -> dict:
    subject = ''
    body = ''
    try:
        if '===이메일 제목===' in crew_output:
            subject_start = crew_output.index('===이메일 제목===') + len('===이메일 제목===')
            subject_end   = crew_output.index('===이메일 제목 끝===')
            subject = crew_output[subject_start:subject_end].strip()
        if '===이메일 본문===' in crew_output:
            body_start = crew_output.index('===이메일 본문===') + len('===이메일 본문===')
            body_end   = crew_output.index('===이메일 본문 끝===')
            body = crew_output[body_start:body_end].strip()
    except ValueError:
        subject = '[뉴스레터] AI 뉴스 인사이트 리포트'
        body = crew_output
    return {'subject': subject, 'body': body}


def send_gmail(to_email: str, subject: str, body: str) -> bool:
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = GMAIL_ADDRESS
        msg['To']      = to_email

        html_body = body.replace('\n', '<br>')
        html_content = f"""
        <html><body>
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: #f8f9fa; border-left: 4px solid #0066cc; padding: 15px; margin-bottom: 20px;">
                <h2 style="color: #0066cc; margin: 0;">📰 AI 뉴스 인사이트</h2>
            </div>
            <div style="line-height: 1.8; color: #333;">
                {html_body}
            </div>
            <div style="margin-top: 30px; padding-top: 15px; border-top: 1px solid #eee;
                        font-size: 12px; color: #888;">
                본 메일은 AI 에이전트(CrewAI + GPT)가 자동 생성한 뉴스레터입니다.
            </div>
        </div>
        </body></html>
        """
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        msg.attach(MIMEText(html_content, 'html', 'utf-8'))

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(GMAIL_ADDRESS, GMAIL_APP_PW)
            server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())

        print(f'✅ 이메일 발송 성공 → {to_email}')
        return True

    except smtplib.SMTPAuthenticationError:
        print('❌ Gmail 인증 실패! 앱 비밀번호를 확인하세요.')
        return False
    except Exception as e:
        print(f'❌ 발송 실패: {e}')
        return False


# ── 전체 파이프라인 ───────────────────────────────────────────────────
def run_pipeline(keyword: str, target_reader: str, send_to_email: str = None) -> dict:
    print('=' * 60)
    print('🚀 뉴스 인사이트 자동화 파이프라인 시작')
    print(f'   키워드 : {keyword}')
    print(f'   독자   : {target_reader}')
    print(f'   발송처 : {send_to_email or "발송 생략"}')
    print('=' * 60)

    task_research, task_write = create_tasks(keyword, target_reader)

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task_research, task_write],
        process=Process.sequential,
        verbose=True
    )

    print('\n⚙️  CrewAI 실행 중...\n')
    crew_result = crew.kickoff()
    result_text = str(crew_result)
    parsed = parse_email_result(result_text)

    send_success = False
    if send_to_email:
        print(f'\n📧 이메일 발송 중 → {send_to_email}')
        send_success = send_gmail(send_to_email, parsed['subject'], parsed['body'])

    print('\n' + '=' * 60)
    print('✅ 파이프라인 완료!')
    print('=' * 60)

    return {
        'keyword'      : keyword,
        'subject'      : parsed['subject'],
        'body'         : parsed['body'],
        'send_success' : send_success
    }


# ── 실행 진입점 ───────────────────────────────────────────────────────
if __name__ == '__main__':
    # 수동 실행 시 입력창 값 사용, 없으면 아래 기본값 사용
    # ✅ 매일 자동 실행 기본값 → 여기서 수정
    KEYWORD       = os.environ.get('INPUT_KEYWORD')  or '삼성전자 반도체'
    TARGET_READER = os.environ.get('INPUT_READER')   or 'IT 업계 투자자'
    SEND_TO       = GMAIL_ADDRESS

    print(f'\n🔑 실행 설정')
    print(f'   키워드 : {KEYWORD}')
    print(f'   독자   : {TARGET_READER}')

    result = run_pipeline(
        keyword=KEYWORD,
        target_reader=TARGET_READER,
        send_to_email=SEND_TO
    )

    print(f'\n제목: {result["subject"]}')
    print(result['body'])
