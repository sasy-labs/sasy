"""Type and order distinctions in the shared adapter record encoding."""
from collections import OrderedDict

import pytest
from sasy.instrumentation._canonical import encoder


@pytest.fixture(params=[False, True], ids=["strict", "fallback"])
def encoding(request):
    def fallback(value):
        raise TypeError("no rendering")

    return encoder(ValueError, fallback if request.param else None)


def test_ordered_dict_records_preserve_order_type_and_nested_values(encoding):
    values = [
        OrderedDict(a=1, b=2),
        OrderedDict(b=2, a=1),
        {"a": 1, "b": 2},
        {"ordered_dict": [["s:a", 1], ["s:b", 2]]},
        [["s:a", 1], ["s:b", 2]],
        OrderedDict(a=1, b=3),
        OrderedDict(),
        {},
    ]
    records = [encoding.json(encoding.canonical(value)) for value in values]
    assert len(set(records)) == len(values)
    assert encoding.canonical({"outer": OrderedDict(a=OrderedDict(b=b"v"))}) == {
        "s:outer": {"ordered_dict": [["s:a", {"ordered_dict": [["s:b", {"bytes": "dg=="}]]}]]},
    }


def test_ordered_dict_keys_keep_their_types(encoding):
    value = OrderedDict([(1, "integer"), ("1", "string"), (b"1", "bytes"), ((1,), "tuple")])
    entries = encoding.canonical(value)["ordered_dict"]
    assert len({key for key, _ in entries}) == 4
    assert [item for _, item in entries] == ["integer", "string", "bytes", "tuple"]
    changed = OrderedDict(value)
    changed["1"] = "changed"
    assert encoding.canonical(changed) != encoding.canonical(value)


def test_ordered_dict_does_not_admit_unknown_subclasses_or_values(encoding):
    class Unknown(OrderedDict):
        pass

    for value in (Unknown(a=1), OrderedDict(a=object()), OrderedDict([(object(), 1)])):
        with pytest.raises(ValueError, match="cannot record"):
            encoding.canonical(value)
