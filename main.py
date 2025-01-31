import time
from typing import List

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger
from together import Together
import os
import httpx

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_SEARCH_API_URL = "https://www.googleapis.com/customsearch/v1"
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX")

client = Together(api_key=TOGETHER_API_KEY)

def get_answer(query: str):
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {"role": "system", "content": "Ты помощник, который выбирает правильный ответ на вопросы об Университете ИТМО."},
                {"role": "user", "content": f"Вопрос: {query}\nВыбери правильный ответ из предложенных вариантов и укажи только его номер (цифру от 1 до 10)."}
            ]
        )

        if response and response.choices:
            answer_num = response.choices[0].message.content.strip().split()[-1]
            return int(answer_num)

    except Exception as e:
        print(f"Ошибка при запросе к DeepSeek API: {e}")
        return -1

def search_links(query: str):
    params = {"key": GOOGLE_SEARCH_API_KEY, "cx": GOOGLE_SEARCH_CX, "q": query}
    try:
        response = httpx.get(GOOGLE_SEARCH_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        links = []
        reasoning_text = "Результаты поиска указывают на следующие источники:"
        for item in data.get("items", [])[:3]:
            title = item.get("title")
            link = item.get("link")
            links.append(HttpUrl(link))
            reasoning_text += f"\n- {title}: {link}"
        return links, reasoning_text
    except Exception as e:
        print(f"Ошибка при поиске с использованием Google Search API: {e}")
        return [], "Не удалось найти релевантную информацию."

app = FastAPI()
logger = None

@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    await logger.info(
        f"Получен запрос: {request.method} {request.url}\n"
        f"Тело запроса: {body.decode()}"
    )

    response = await call_next(request)
    duration = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Запрос завершен: {request.method} {request.url}\n"
        f"Статус: {response.status_code}\n"
        f"Тело ответа: {response_body.decode()}\n"
        f"Время выполнения: {duration:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        await logger.info(f"Обработка запроса предсказания с id: {body.id}")
        answer = get_answer(body.query)
        sources, reasoning = search_links(body.query)

        response_data = PredictionResponse(
            id=body.id,
            answer=answer,
            reasoning=reasoning,
            sources=sources,
        )
        await logger.info(f"Запрос {body.id} успешно обработан")
        return response_data
    except ValueError as e:
        error_message = str(e)
        await logger.error(f"Ошибка валидации для запроса {body.id}: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)
    except Exception as e:
        await logger.error(f"Внутренняя ошибка при обработке запроса {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")
