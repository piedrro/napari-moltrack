"""Captured cell-tool inputs and freshness-checked detached results."""

from copy import deepcopy
from dataclasses import dataclass
from weakref import ref

from molcore import StaleResultError
from molcore.staging import state_token, validate_expected

from .model_processing import _cancel


@dataclass(frozen=True)
class PreparedCells:
    payload: object
    expected: object
    owner: object
    cancel_event: object = None


@dataclass(frozen=True)
class PendingCells:
    prepared: PreparedCells
    value: object


def prepare_cells(session, payload, *, cancel_event=None):
    return PreparedCells(
        deepcopy(payload),
        state_token(session.model),
        ref(session.model),
        cancel_event,
    )


def finish_cells(prepared, value):
    _cancel(prepared.cancel_event)
    return PendingCells(prepared, value)


def validate_cells(session, pending):
    _cancel(pending.prepared.cancel_event)
    if session.model is not pending.prepared.owner():
        raise StaleResultError("Session model changed")
    validate_expected(session.model, pending.prepared.expected)
    return pending.value
