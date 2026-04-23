from shared.models import Spec

def validate_spec(data):
    try:
        return Spec(**data).model_dump()
    except Exception:
        return None
