from rtrec.privacy import pseudonymize


def test_pseudonym_is_stable_for_same_key_and_id():
    assert pseudonymize("u1", key=b"k") == pseudonymize("u1", key=b"k")


def test_pseudonym_changes_with_key_and_id():
    assert pseudonymize("u1", key=b"k") != pseudonymize("u1", key=b"other")
    assert pseudonymize("u1", key=b"k") != pseudonymize("u2", key=b"k")


def test_pseudonym_does_not_contain_the_raw_id():
    assert "u1234" not in pseudonymize("u1234", key=b"k")
