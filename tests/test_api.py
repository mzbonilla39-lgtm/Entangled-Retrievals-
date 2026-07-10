import pytest
from httpx import AsyncClient
from fastapi import status


@pytest.mark.asyncio
async def test_index_and_retrieve():
    from src.main import app

    docs = [
        {"id": "d1", "text": "The quick brown fox"},
        {"id": "d2", "text": "A slow yellow dog"},
    ]

    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.post("/index", json=docs)
        assert r.status_code == status.HTTP_200_OK
        assert r.json().get("indexed") == 2

        r2 = await ac.post("/retrieve", json={"q": "quick fox", "k": 2})
        assert r2.status_code == status.HTTP_200_OK
        data = r2.json()
        assert data["query"] == "quick fox"
        assert len(data["results"]) >= 1
