import os
import anthropic

# API 키 설정
# 환경변수로 설정 권장: export ANTHROPIC_API_KEY="your_key"
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def parse_goal(user_command: str) -> str:
    """
    자연어 명령을 goal_text로 변환
    예시:
        "의자로 가줘"           → "a chair"
        "저기 빨간 의자 찾아줘"  → "a red chair"
        "문 찾아서 통과해줘"     → "a door"
        "사람 따라가줘"          → "a person"

    returns: 영어 goal_text 문자열
    """
    if not user_command.strip():
        return ""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": f"""다음 명령에서 로봇이 찾아가야 할 목표 물체를 영어로 짧게 추출해줘.

규칙:
- 영어로만 출력
- 관사(a/an/the) 포함
- 색상이나 특징이 있으면 포함
- 다른 설명 없이 목표 물체만 출력

예시:
"의자로 가줘" → a chair
"빨간 의자 찾아줘" → a red chair
"문 통과해줘" → a door
"사람 따라가줘" → a person
"테이블 옆으로 가줘" → a table

명령: {user_command}
출력:"""
                }
            ]
        )
        result = message.content[0].text.strip().lower()
        # 혹시 따옴표 포함되면 제거
        result = result.replace('"', '').replace("'", "")
        return result

    except Exception as e:
        print(f"[llm_parser] LLM 호출 실패: {e}")
        # 실패 시 원본 입력 반환
        return user_command.strip()


def parse_goal_simple(user_command: str) -> str:
    """
    LLM 없이 간단한 키워드 매핑으로 목표 추출 (오프라인 fallback)
    """
    keyword_map = {
        "의자": "a chair",
        "chair": "a chair",
        "문": "a door",
        "door": "a door",
        "테이블": "a table",
        "table": "a table",
        "사람": "a person",
        "person": "a person",
        "병": "a bottle",
        "bottle": "a bottle",
        "컵": "a cup",
        "cup": "a cup",
        "소파": "a sofa",
        "sofa": "a sofa",
        "책상": "a desk",
        "desk": "a desk",
    }

    lower = user_command.lower()
    for keyword, goal in keyword_map.items():
        if keyword in lower:
            return goal

    # 매핑 실패 시 원본 반환
    return user_command.strip()