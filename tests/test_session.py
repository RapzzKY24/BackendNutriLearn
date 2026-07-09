from app.services.rag_service import SessionManager


def test_session_create():
    sm = SessionManager(max_turns=6)
    history = sm.get_history("session-1")
    assert history == []


def test_session_add_turn():
    sm = SessionManager(max_turns=6)
    sm.add_turn("session-1", "Halo", "Hai, ada yang bisa dibantu?")
    history = sm.get_history("session-1")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_session_multiple_turns():
    sm = SessionManager(max_turns=6)
    sm.add_turn("s1", "q1", "a1")
    sm.add_turn("s1", "q2", "a2")
    sm.add_turn("s1", "q3", "a3")
    history = sm.get_history("s1")
    assert len(history) == 6
    assert history[-1]["content"] == "a3"


def test_session_max_turns():
    sm = SessionManager(max_turns=2)
    sm.add_turn("s1", "q1", "a1")
    sm.add_turn("s1", "q2", "a2")
    sm.add_turn("s1", "q3", "a3")
    history = sm.get_history("s1")
    assert len(history) == 4
    assert history[0]["content"] == "q2"


def test_session_empty_id():
    sm = SessionManager(max_turns=6)
    sm.add_turn("", "q1", "a1")
    history = sm.get_history("")
    assert history == []


def test_session_flush():
    sm = SessionManager(max_turns=6)
    sm.add_turn("s1", "q1", "a1")
    sm.flush("s1")
    assert sm.get_history("s1") == []


def test_session_flush_all():
    sm = SessionManager(max_turns=6)
    sm.add_turn("s1", "q1", "a1")
    sm.add_turn("s2", "q1", "a1")
    sm.flush()
    assert sm.get_history("s1") == []
    assert sm.get_history("s2") == []
