import typing
import types
from typing import Any, Dict, Tuple, Type, Union
import ast

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo



def ensure_signature(sig: str | BaseModel) -> BaseModel:
    if sig is None:
        return None
    if isinstance(sig, str):
        return signature(sig)
    return sig


def signature(
    signature: Union[str, Dict[str, Tuple[type, FieldInfo]]],
    signature_name: str = "Answer",
) -> Type[BaseModel]:
    """Create a new Signature type with the given fields and instructions.

    Note:
        Even though we're calling a type, we're not making an instance of the type.
        In general, instances of Signature types are not allowed to be made. The call
        syntax is provided for convenience.

    Args:
        signature: The signature format, specified as "output1:type1, output2:type2".
        instructions: An optional prompt for the signature.
        signature_name: An optional name for the new signature type.
    """
    fields = _parse_signature(signature) if isinstance(signature, str) else signature

    # Validate the fields, this is important because we sometimes forget the
    # slightly unintuitive syntax with tuples of (type, Field)
    fixed_fields = {}
    for name, type_field in fields.items():
        if not isinstance(name, str):
            raise ValueError(f"Field names must be strings, not {type(name)}")
        if isinstance(type_field, FieldInfo):
            type_ = type_field.annotation
            field = type_field
        else:
            if not isinstance(type_field, tuple):
                raise ValueError(f"Field values must be tuples, not {type(type_field)}")
            type_, field = type_field
        # It might be better to be explicit about the type, but it currently would break
        # program of thought and teleprompters, so we just silently default to string.
        if type_ is None:
            type_ = str
        # if not isinstance(type_, type) and not isinstance(typing.get_origin(type_), type):
        if not isinstance(type_, (type, typing._GenericAlias, types.GenericAlias)):
            raise ValueError(f"Field types must be types, not {type(type_)}")
        if not isinstance(field, FieldInfo):
            raise ValueError(f"Field values must be Field instances, not {type(field)}")
        fixed_fields[name] = (type_, field)

    # Fixing the fields shouldn't change the order
    assert list(fixed_fields.keys()) == list(fields.keys())  # noqa: S101

    return create_model(
        signature_name,
        # __base__=BaseModel,
        **fixed_fields,
    )


def _parse_signature(input_str: str) -> Tuple[Type, Field]:
    return {name: (type_, Field()) for name, type_ in _parse_arg_string(input_str)}


def _parse_arg_string(string: str, names=None) -> Dict[str, str]:
    args = ast.parse("def f(" + string + "): pass").body[0].args.args
    names = [arg.arg for arg in args]
    types = [str if arg.annotation is None else _parse_type_node(arg.annotation) for arg in args]
    return zip(names, types)


def _parse_type_node(node, names=None) -> Any:
    """Recursively parse an AST node representing a type annotation.

    without using structural pattern matching introduced in Python 3.10.
    """
    if names is None:
        names = typing.__dict__

    if isinstance(node, ast.Module):
        body = node.body
        if len(body) != 1:
            raise ValueError(f"Code is not syntactically valid: {node}")
        return _parse_type_node(body[0], names)

    if isinstance(node, ast.Expr):
        value = node.value
        return _parse_type_node(value, names)

    if isinstance(node, ast.Name):
        id_ = node.id
        if id_ in names:
            return names[id_]
        for type_ in [int, str, float, bool, list, tuple, dict]:
            if type_.__name__ == id_:
                return type_
        raise ValueError(f"Unknown name: {id_}")

    if isinstance(node, ast.Subscript):
        base_type = _parse_type_node(node.value, names)
        arg_type = _parse_type_node(node.slice, names)
        return base_type[arg_type]

    if isinstance(node, ast.Tuple):
        elts = node.elts
        return tuple(_parse_type_node(elt, names) for elt in elts)

    if isinstance(node, ast.Call) and node.func.id == "Field":
        keys = [kw.arg for kw in node.keywords]
        values = [kw.value.value for kw in node.keywords]
        return Field(**dict(zip(keys, values)))

    raise ValueError(f"Code is not syntactically valid: {node}")
