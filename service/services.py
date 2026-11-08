"""Use cases shared by the JSON API and the UI. No HTTP here and no SQL (that lives in repository)."""
import repository
from crew.bundle import Bundle
from crew.scoring import score_record


class ClientNotFound(LookupError):
    """The client has no stored feature record (unknown id, or loaded without features)."""


def score_client(bundle: Bundle, sk_id_curr: int, source: str) -> dict:
    """Score a stored client and persist the result. RecordError (bad stored data) propagates to the caller."""
    features = repository.get_features(sk_id_curr)
    if features is None:
        raise ClientNotFound(f"client {sk_id_curr} has no stored features")
    result = score_record(bundle, features, sk_id_curr)
    saved = repository.save_score(sk_id_curr, result, source)
    return {**result, **saved}
