import fakeredis
import pytest

from rtrec.cache import RedisCache
from rtrec.service import Recommender


@pytest.fixture()
def recommender(artifacts):
    return Recommender.load(artifacts, cache=RedisCache(fakeredis.FakeRedis()), ttl=60)


def test_known_user_gets_personalized_k_items(recommender):
    out = recommender.recommend("u3", k=5)
    assert out["source"] == "personalized"
    assert len(out["items"]) == 5
    scores = [item["score"] for item in out["items"]]
    assert scores == sorted(scores, reverse=True)


def test_recommendations_exclude_items_the_user_already_played(recommender):
    seen = {f"i{i}" for i in recommender.history.seen(3).tolist()}
    out = recommender.recommend("u3", k=10)
    assert not seen & {item["item_id"] for item in out["items"]}


def test_unknown_user_gets_cold_start_popularity(recommender):
    out = recommender.recommend("new-listener", k=4)
    assert out["source"] == "cold_start"
    assert len(out["items"]) == 4
    assert len({item["item_id"] for item in out["items"]}) == 4


def test_second_request_is_served_from_cache(recommender):
    first = recommender.recommend("u7", k=3)
    second = recommender.recommend("u7", k=3)
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["items"] == first["items"]


def test_service_works_without_cache(artifacts):
    rec = Recommender.load(artifacts)
    assert rec.recommend("u1", k=2)["cache_hit"] is False
