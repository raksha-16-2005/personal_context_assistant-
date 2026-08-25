from __future__ import annotations

import base64
import json
import stat
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.corpus import gmail as G

RAW = (b"From: sara@company.com\r\n"
       b"To: mark@company.com\r\n"
       b"Subject: MSA redline\r\n"
       b"Date: Tue, 30 Oct 2001 09:12:00 -0600\r\n"
       b"Message-ID: <abc@company.com>\r\n\r\n"
       b"Can you send comments before Thursday?\r\n")


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


# -- tokens on disk ---------------------------------------------------------

def test_the_token_file_is_written_owner_only(tmp_path):
    # This token can read an entire mailbox. It must never be group- or
    # world-readable, even briefly.
    store = G.TokenStore(path=tmp_path / "t.json")
    store.data = {"refresh_token": "secret", "access_token": "a",
                  "expires_at": time.time() + 3600, "issued_at": time.time()}
    store.save()

    mode = stat.S_IMODE((tmp_path / "t.json").stat().st_mode)
    assert mode == 0o600


def test_the_default_token_path_is_outside_the_repo():
    # .gitignore is a convention, and `git add -f` or a moved path defeats it. A
    # mailbox-reading token should not be one mistake from a public commit.
    assert ".config" in str(G.DEFAULT_TOKEN_PATH)
    assert "personal-context-assistant" not in str(G.DEFAULT_TOKEN_PATH)


def test_tokens_roundtrip(tmp_path):
    path = tmp_path / "t.json"
    store = G.TokenStore(path=path)
    store.data = {"refresh_token": "r", "access_token": "a", "expires_at": 123}
    store.save()

    assert G.TokenStore(path=path).load().refresh_token == "r"


def test_a_missing_token_file_is_not_an_error(tmp_path):
    assert G.TokenStore(path=tmp_path / "absent.json").load().refresh_token == ""


def test_an_expired_access_token_is_not_considered_valid(tmp_path):
    store = G.TokenStore(path=tmp_path / "t.json")
    store.data = {"access_token": "a", "expires_at": time.time() - 1}
    assert not store.access_token_valid()


def test_a_token_expiring_within_the_skew_is_not_valid(tmp_path):
    # Treating a token with 10 seconds left as valid means a request that fails
    # mid-flight instead of a refresh that succeeds.
    store = G.TokenStore(path=tmp_path / "t.json")
    store.data = {"access_token": "a", "expires_at": time.time() + 10}
    assert not store.access_token_valid(skew=60)


# -- the 7-day refresh token clock ------------------------------------------

def test_the_seven_day_limit_is_reported_before_it_bites(tmp_path):
    # Apps in Testing get 7-day refresh tokens for Gmail scopes. The failure
    # otherwise arrives as an opaque invalid_grant in the middle of a sync.
    store = G.TokenStore(path=tmp_path / "t.json")
    store.data = {"refresh_token": "r", "issued_at": time.time() - 6.5 * 86400}

    assert store.days_until_reauthorization() == pytest.approx(0.5, abs=0.05)
    assert "expires in 0.5 days" in store.warn_if_expiring()


def test_an_expired_refresh_token_says_what_to_do(tmp_path):
    store = G.TokenStore(path=tmp_path / "t.json")
    store.data = {"refresh_token": "r", "issued_at": time.time() - 8 * 86400}

    warning = store.warn_if_expiring()
    assert "older than 7 days" in warning
    assert "re-run the authorization" in warning


def test_a_fresh_token_produces_no_warning(tmp_path):
    store = G.TokenStore(path=tmp_path / "t.json")
    store.data = {"refresh_token": "r", "issued_at": time.time()}
    assert store.warn_if_expiring() == ""


def test_the_documented_limit_matches_googles_behaviour():
    assert G.TESTING_REFRESH_TOKEN_DAYS == 7


# -- the authorization URL --------------------------------------------------

def test_the_consent_url_asks_for_offline_access_and_forces_consent():
    # access_type=offline is what yields a refresh token at all; without
    # prompt=consent Google omits it on re-authorisation.
    url = G.authorization_url("client-123")

    assert "access_type=offline" in url
    assert "prompt=consent" in url
    assert "gmail.readonly" in url
    assert "client_id=client-123" in url


def test_the_scope_is_readonly():
    # There is no narrower Gmail scope that still allows search.
    assert G.SCOPE.endswith("gmail.readonly")


def test_extra_scopes_can_be_requested_for_the_web_app():
    # The web app's calendar-suggestions flow needs calendar.events on top of
    # the desktop flow's gmail.readonly-only consent screen.
    url = G.authorization_url("client-123", scopes=f"{G.SCOPE} {G.CALENDAR_SCOPE}")

    assert "gmail.readonly" in url
    assert "calendar.events" in url


def test_the_desktop_flow_still_defaults_to_gmail_only():
    url = G.authorization_url("client-123")

    assert "gmail.readonly" in url
    assert "calendar.events" not in url


# -- the client -------------------------------------------------------------

def test_missing_credentials_name_every_setup_step():
    with pytest.raises(G.NeedsAuthorization) as exc:
        G.GmailClient("", "")

    message = str(exc.value)
    assert "Gmail API" in message
    assert "Testing" in message
    assert "gmail_auth.py" in message


def _client(tmp_path, tokens=None):
    store = G.TokenStore(path=tmp_path / "t.json")
    store.data = tokens if tokens is not None else {
        "access_token": "a", "expires_at": time.time() + 3600,
        "refresh_token": "r", "issued_at": time.time()}
    store.save()
    return G.GmailClient("cid", "secret", tokens=G.TokenStore(path=store.path))


def test_a_client_with_no_refresh_token_says_how_to_get_one(tmp_path):
    client = _client(tmp_path, tokens={})
    with pytest.raises(G.NeedsAuthorization, match="gmail_auth"):
        client._bearer()


def test_raw_messages_are_decoded_from_url_safe_base64(tmp_path, monkeypatch):
    client = _client(tmp_path)
    monkeypatch.setattr(client, "_get", lambda *a, **k: {"raw": _b64(RAW)})

    assert client.raw_message("id1") == RAW


def test_a_message_with_no_payload_is_an_error_not_empty_bytes(tmp_path, monkeypatch):
    client = _client(tmp_path)
    monkeypatch.setattr(client, "_get", lambda *a, **k: {})

    with pytest.raises(G.GmailError, match="no raw payload"):
        client.raw_message("id1")


def test_listing_follows_pagination(tmp_path, monkeypatch):
    client = _client(tmp_path)
    pages = [
        {"messages": [{"id": "a"}, {"id": "b"}], "nextPageToken": "p2"},
        {"messages": [{"id": "c"}]},
    ]
    monkeypatch.setattr(client, "_get", lambda *a, **k: pages.pop(0))

    assert list(client.list_message_ids()) == ["a", "b", "c"]


def test_listing_respects_max_messages(tmp_path, monkeypatch):
    client = _client(tmp_path)
    monkeypatch.setattr(client, "_get", lambda *a, **k: {
        "messages": [{"id": str(i)} for i in range(500)], "nextPageToken": "next"})

    assert len(list(client.list_message_ids(max_messages=3))) == 3


def test_history_yields_added_message_ids(tmp_path, monkeypatch):
    client = _client(tmp_path)
    monkeypatch.setattr(client, "_get", lambda *a, **k: {
        "history": [{"messagesAdded": [{"message": {"id": "x"}},
                                       {"message": {"id": "y"}}]},
                    {"messagesAdded": [{"message": {}}]}]})

    assert list(client.history_since("100")) == ["x", "y"]


# -- one parser for both corpora --------------------------------------------

def test_gmail_messages_go_through_the_enron_parser(tmp_path, monkeypatch):
    # The whole point: Gmail returns RFC822, so both corpora use exactly one
    # parser. A second parser would quietly produce incomparable rows - different
    # body extraction, a different dedup key for the same message.
    from emailrag.corpus.enron import parse_message

    client = _client(tmp_path)
    monkeypatch.setattr(client, "_get", lambda *a, **k: {"raw": _b64(RAW)})
    monkeypatch.setattr(client, "list_message_ids",
                        lambda *a, **k: iter(["id1"]))

    rows = G.fetch_messages(client)

    assert len(rows) == 1
    assert rows[0]["sender"] == "sara@company.com"
    assert rows[0]["subject"] == "MSA redline"
    assert rows[0]["source_path"] == "gmail:id1"
    # Byte-identical parsing means an identical dedup key.
    assert rows[0]["dedup_key"] == parse_message(RAW).dedup_key


def test_one_unreadable_message_does_not_end_the_sync(tmp_path, monkeypatch):
    client = _client(tmp_path)
    monkeypatch.setattr(client, "list_message_ids", lambda *a, **k: iter(["bad", "good"]))

    def raw(message_id):
        if message_id == "bad":
            raise G.GmailError("410 gone")
        return RAW

    monkeypatch.setattr(client, "raw_message", raw)

    rows = G.fetch_messages(client)
    assert len(rows) == 1        # the good one survived


def test_on_progress_reports_the_exact_total_before_fetching_anything(tmp_path, monkeypatch):
    client = _client(tmp_path)
    monkeypatch.setattr(client, "list_message_ids",
                        lambda *a, **k: iter(["id1", "id2", "id3"]))
    monkeypatch.setattr(client, "raw_message", lambda message_id: RAW)

    calls = []
    G.fetch_messages(client, on_progress=lambda completed, total: calls.append((completed, total)))

    # The very first call reports the real total (3), before any message has
    # actually been fetched - a caller showing "X of Y" needs Y from the start.
    assert calls[0] == (0, 3)
    # And the last call reflects every message actually completed, even
    # though 3 is not a multiple of the 50-per-update cadence.
    assert calls[-1] == (3, 3)


def test_on_progress_final_call_lands_even_under_the_batch_size(tmp_path, monkeypatch):
    # Regression check for the "stuck a few messages short of done" gap: a
    # fetch smaller than 50 messages only ever hits the unconditional final
    # call, never the `completed % 50 == 0` one.
    client = _client(tmp_path)
    monkeypatch.setattr(client, "list_message_ids", lambda *a, **k: iter(["id1"]))
    monkeypatch.setattr(client, "raw_message", lambda message_id: RAW)

    calls = []
    G.fetch_messages(client, on_progress=lambda completed, total: calls.append((completed, total)))

    assert (1, 1) in calls


# -- incremental sync -------------------------------------------------------

def test_sync_state_roundtrips(tmp_path):
    path = tmp_path / "state.json"
    G.SyncState(history_id="500", last_sync_utc="2026-07-27T00:00:00Z",
                messages_seen=10).save(path)

    state = G.SyncState.load(path)
    assert state.history_id == "500" and state.messages_seen == 10


def test_a_first_sync_is_full_and_records_the_cursor(tmp_path, monkeypatch):
    client = _client(tmp_path)
    monkeypatch.setattr(client, "current_history_id", lambda: "900")
    monkeypatch.setattr(G, "fetch_messages", lambda *a, **k: [{"dedup_key": "k"}])

    messages, state = G.sync(client, tmp_path / "state.json")

    assert len(messages) == 1
    assert state.history_id == "900"


def test_a_second_sync_uses_history_instead_of_relisting(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    G.SyncState(history_id="500").save(state_path)

    client = _client(tmp_path)
    monkeypatch.setattr(client, "current_history_id", lambda: "900")
    monkeypatch.setattr(client, "history_since", lambda h: iter(["new1"]))
    monkeypatch.setattr(client, "raw_message", lambda mid: RAW)
    monkeypatch.setattr(G, "fetch_messages",
                        lambda *a, **k: pytest.fail("relisted the whole mailbox"))

    messages, state = G.sync(client, state_path)

    assert len(messages) == 1
    assert state.history_id == "900"


def test_sync_reports_progress_on_the_incremental_path_too(tmp_path, monkeypatch):
    # The incremental path fetches serially, not through fetch_messages - it
    # has to report on_progress itself rather than inheriting it for free.
    state_path = tmp_path / "state.json"
    G.SyncState(history_id="500").save(state_path)

    client = _client(tmp_path)
    monkeypatch.setattr(client, "current_history_id", lambda: "900")
    monkeypatch.setattr(client, "history_since", lambda h: iter(["new1", "new2"]))
    monkeypatch.setattr(client, "raw_message", lambda mid: RAW)

    calls = []
    G.sync(client, state_path, on_progress=lambda completed, total: calls.append((completed, total)))

    assert calls[0] == (0, 2)     # total known before any fetch
    assert calls[-1] == (2, 2)    # ends on the real final count


def test_an_expired_cursor_falls_back_to_a_full_sync(tmp_path, monkeypatch):
    # Google keeps roughly a week of history. A cursor older than that is a normal
    # event after time away, not an error.
    state_path = tmp_path / "state.json"
    G.SyncState(history_id="1").save(state_path)

    client = _client(tmp_path)
    monkeypatch.setattr(client, "current_history_id", lambda: "900")

    def expired(_):
        raise G.HistoryTooOld("too old")
        yield  # pragma: no cover

    monkeypatch.setattr(client, "history_since", expired)
    monkeypatch.setattr(G, "fetch_messages", lambda *a, **k: [{"dedup_key": "k"}])

    messages, state = G.sync(client, state_path)

    assert len(messages) == 1
    assert state.history_id == "900"


def test_the_cursor_is_read_before_fetching_so_nothing_is_skipped(tmp_path, monkeypatch):
    # Messages arriving during a sync must be picked up next time, not fall into a
    # gap between "fetched" and "cursor advanced".
    calls = []
    client = _client(tmp_path)
    monkeypatch.setattr(client, "current_history_id",
                        lambda: calls.append("cursor") or "900")
    monkeypatch.setattr(G, "fetch_messages",
                        lambda *a, **k: calls.append("fetch") or [])

    G.sync(client, tmp_path / "state.json")

    assert calls == ["cursor", "fetch"]


# -- the extraction window --------------------------------------------------

def test_the_since_query_is_a_gmail_search_expression():
    # Retrieval wants all history; extraction does not. At ~25 s/message a
    # 35k-message mailbox is ~240 hours, so extraction is capped at a window.
    q = G.since_query(90)
    assert q.startswith("after:")
    assert len(q.split("after:")[1].split("/")) == 3


def test_the_before_query_is_the_complementary_half_of_since_query():
    # The staged first sync (webapp) fetches `since_query(N)` fast, then
    # backfills `before_query(N)` in the background - the same cutoff date,
    # the opposite direction, so together they cover the whole mailbox with
    # no overlap and no gap.
    since_cutoff = G.since_query(30).split("after:")[1]
    before_cutoff = G.before_query(30).split("before:")[1]
    assert since_cutoff == before_cutoff
