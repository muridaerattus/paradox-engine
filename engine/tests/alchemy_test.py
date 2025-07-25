from alchemy.operations import alchemy_and, alchemy_or

def test_alchemy_and():
    # Canonical example from the comic!
    code1 = "nZ7Un6BI"
    code2 = "DQMmJLeK"
    result = alchemy_and(code1, code2)
    assert result == "126GH48G"

def test_alchemy_and_trivial():
    code1 = "nZ7Un6BI"
    code2 = "00000000"
    result = alchemy_and(code1, code2)
    assert result == "00000000"

def test_alchemy_or_trivial():
    code1 = "nZ7Un6BI"
    code2 = "!!!!!!!!"
    result = alchemy_or(code1, code2)
    assert result == "!!!!!!!!"