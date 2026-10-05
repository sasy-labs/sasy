"""The canonical encoding an adapter writes a message down with.

``model_dump(mode="json")`` is lossy, and every loss is two messages sharing one
record. This module is that encoding in one place, so the adapters that record
messages agree on it. Each adapter builds its own copy with :func:`encoder`,
naming the error type it raises, because a value this encoding cannot write down
is a stopped run of that adapter and has to read as one.

What a record guarantees: it covers every field of the value it is given — no
part of that value is left out of the record — and any two values that differ
as JSON data are written as two different records.

Two values that are *equal* as JSON data and differ only in their Python type —
a tuple and the list it writes as, bytes and the text they decode to, a
character and the surrogate pair JSON spells it with, a typed object and the
dict it dumps to — are told apart where this encoding covers the type, which is
the list in :func:`encoder`. Beyond that list such a pair is outside the
guarantee, and it is outside it safely: under this project's trust model the
program's own code is trusted while the data an agent reads and the model's
output are not, and no untrusted value can be one half of such a pair. Data and
model output reach the program as text or JSON; JSON has no tuples, no bytes, no
enum members and no model instances, and ``json.loads`` turns a surrogate-pair
escape into the one character it stands for. Only the trusted program can
produce the other half.
"""

from __future__ import annotations

import json
from base64 import b64encode
from collections.abc import Callable
from enum import Enum
from typing import Any, NamedTuple


class Encoder(NamedTuple):
    """The three functions an adapter imports under its own private names."""

    json: Callable[[Any], str]
    canonical: Callable[[Any], Any]
    canonical_key: Callable[[Any], str]


def _json(value: Any) -> str:
    # Reject opaque runtime objects, NaN, and lossy fallback stringification.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def encoder(error: type[Exception], fallback: Callable[[Any], Any] | None = None) -> Encoder:
    """The canonical encoding, refusing what it cannot write down with *error*.

    The encoding is injective over the types it covers: JSON's own types, bytes,
    bytearray, tuple, set, frozenset, and — under a fallback — enum members. Two
    different values of those types never share one encoding.

    An adapter that records values it does not choose — an ADK tool's arguments
    and its return are arbitrary Python — gives a *fallback*: a rendering of a
    value of a type this encoding does not cover. The value is then written as
    that rendering under a wrapper naming its type's module and qualified name,
    rather than stopping the run. That wrapper tells the value apart from every
    natively encoded value and from the values of types with another name. It is
    **not** a guarantee that two values of that one type differ — a rendering is
    as lossy as the fallback makes it — nor that two classes built with the same
    module and qualified name differ. An adapter that leaves *fallback* out keeps
    the stricter boundary: a type this encoding does not cover stops the run, by
    name. Either way the module docstring's guarantee holds: what the wrapper
    does not separate is a pair only the trusted program can build.
    """

    def canonical_key(key: Any) -> str:
        """A dict key rendered as a string, keeping the kinds of key apart.

        JSON has string keys only, so every other kind of key has to be written
        as one; each kind gets a prefix of its own, and no two prefixes are a
        prefix of each other. Two keys of the types :func:`canonical` covers
        therefore never render alike; a key of any other type renders as that
        function writes it, with the same narrower guarantee. This is also what
        makes the wrapper keys in
        :func:`canonical` safe: every key of an encoded dict starts with one of
        those prefixes — ``s:``, ``bool:``, ``i:``, ``b:``, ``n:``, ``o:`` — and
        no wrapper name does: ``bytes``, ``bytearray``, ``tuple``, ``set``,
        ``frozenset``, ``surrogates`` and ``py:<module>:<qualname>`` all start
        with something else. So no dict can be written to look like a wrapper.

        A key of a type the encoding does not cover reaches ``canonical`` under
        the ``o:`` prefix and is handled there like any other value: written
        under its type's wrapper when the adapter gave a fallback, refused by
        name when it did not.
        The kinds named here are the hashable ones; a key is hashable by
        construction, so ``bytearray`` and ``set`` — which :func:`canonical`
        does keep apart from ``bytes`` and ``frozenset`` as values — cannot
        appear in this position.
        """
        # Exact types, for the reason given in :func:`canonical`: a str subclass
        # would take the ``s:`` prefix and land on the plain string's key.
        if type(key) is str:
            rendered = canonical(key)
            # A string that is not valid Unicode is written as a wrapper rather
            # than as a string, so it takes the ``o:`` prefix, which a plain
            # string key never produces.
            return "s:" + rendered if type(rendered) is str else "o:" + _json(rendered)
        if type(key) is bool:
            return "bool:true" if key else "bool:false"
        if type(key) is int:
            return "i:" + str(key)
        if type(key) is bytes:
            return "b:" + b64encode(key).decode()
        if key is None:
            return "n:"
        return "o:" + _json(canonical(key))

    def canonical(value: Any) -> Any:
        """*value* as JSON data, keeping apart everything JSON would run together.

        ``model_dump(mode="json")`` is lossy, and every loss is two messages
        sharing one record: it writes ``b"k"`` and ``"k"`` as the same string, a
        tuple and a set as the same list, a UUID or an enum member as its string
        form, and every dict key as a string, so ``{1: x}`` and ``{"1": x}`` come
        out alike. Those are not exotic — a bytes key in a LangChain message's
        ``additional_kwargs`` is the difference between a system message the
        pinned client transmits as ``system`` and one it transmits as
        ``developer``.

        So this works on the Python dump and tags by *type*: bytes, bytearrays,
        tuples, sets and frozensets each become a one-key object of their own
        that no dict can imitate, and the kinds JSON already tells apart are left
        as they are, which keeps a node readable. A type it does not know is
        never rendered as one it does: with a *fallback* it becomes a wrapper
        named after the type's module and qualified name, and without one it is
        refused. So a value of an unanticipated type never collides with a value
        of a type this encoding covers, nor with a value of a differently named
        type; whether two values of that one type differ is up to the fallback's
        rendering, which this encoding does not control. Enum members are the
        one such type it makes exact, by writing the member's name alongside the
        rendering of its value.
        """
        # Exact types, not isinstance: a str subclass such as a string enum is a
        # str to isinstance and writes as its value, which is the plain string's
        # record. A subclass is a type the encoding has not anticipated, so it
        # takes the same path as any other — refused, by name.
        if value is None or type(value) in (bool, str):
            if type(value) is str:
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError as unicode_error:
                    # A lone surrogate is not text: JSON escapes a surrogate
                    # pair exactly as it escapes the character that pair stands
                    # for, so the two share one record. The bytes the string
                    # keeps are what tells them apart; an adapter that has not
                    # asked for a total record stops here instead.
                    if fallback is None:
                        raise error(
                            "A message holds text that is not valid Unicode, which has no "
                            "faithful record") from unicode_error
                    return {"surrogates": b64encode(value.encode("utf-8", "surrogatepass")).decode()}
            return value
        if type(value) is int:
            return value
        if type(value) is float:
            if value != value or value in (float("inf"), float("-inf")):
                raise error(
                    "A message holds a non-finite number, which has no faithful record")
            return value
        # Each type gets its own wrapper: two types sharing one would be two
        # messages sharing one record.
        if type(value) is bytes:
            return {"bytes": b64encode(value).decode()}
        if type(value) is bytearray:
            return {"bytearray": b64encode(bytes(value)).decode()}
        if type(value) is list:
            return [canonical(item) for item in value]
        if type(value) is tuple:
            return {"tuple": [canonical(item) for item in value]}
        if type(value) in (set, frozenset):
            # A set has no order of its own, so it gets the one order its own
            # encoding gives: distinct sets stay distinct, equal ones stay equal.
            name = "set" if type(value) is set else "frozenset"
            return {name: sorted((canonical(item) for item in value), key=_json)}
        if type(value) is dict:
            return {canonical_key(key): canonical(item) for key, item in value.items()}
        if fallback is not None:
            # The type's module and qualified name name the wrapper, so two
            # differently named types cannot share one record even when they
            # render alike; ``py:`` is not one of the dict-key prefixes, so no
            # encoded dict can produce this wrapper either. The two parts are
            # joined with a colon, which neither a module name nor a qualified
            # name can contain, so one wrapper name has one reading: a dot
            # would let ("pkg.outer", "Value") and ("pkg", "outer.Value") share
            # it.
            if isinstance(value, Enum):
                # A member's identity is its class and its member name, and a
                # fallback renders it as its value alone — which two members can
                # share. The name carried alongside the value's own canonical
                # record makes the member exact, as a value and as a dict key.
                return {f"py:{type(value).__module__}:{type(value).__qualname__}":
                        {"enum": value.name, "value": canonical(value.value)}}
            try:
                rendered = fallback(value)
                if type(rendered) is type(value):
                    raise TypeError("the fallback returned the value itself")
            except Exception as fallback_error:
                raise error(
                    f"A message holds a {type(value).__name__}, which this adapter cannot "
                    "record: it has no rendering at all, not even the one a JSON dump "
                    "would write.") from fallback_error
            return {f"py:{type(value).__module__}:{type(value).__qualname__}": canonical(rendered)}
        raise error(
            f"A message holds a {type(value).__name__}, which this adapter cannot record "
            "faithfully: it records JSON data, bytes, tuples and sets, and writing anything "
            "else as one of those would let two different messages share one record and one "
            "mark. Convert the value before the message is recorded.")

    return Encoder(_json, canonical, canonical_key)
