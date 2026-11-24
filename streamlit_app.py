"""SpamShield - interface Streamlit.

Le frontend ne charge jamais le modèle : il appelle uniquement l'API FastAPI.
L'URL de l'API vient de la variable d'environnement API_URL
(défaut : http://localhost:8000 ; dans Docker Compose : http://api:8000).
"""

import json
import os
import urllib.error
import urllib.request

import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_S = 5
VALID_PREDICTIONS = {"spam", "legitime"}


class ApiError(Exception):
    """L'API a répondu, mais avec une erreur ou une réponse inattendue."""


class ApiUnavailable(ApiError):
    """L'API est injoignable (réseau, DNS, timeout, connexion refusée)."""


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{API_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # doit précéder URLError (sous-classe)
        if exc.code == 422:
            raise ApiError("Texte refusé par l'API (422) : vérifiez le contenu saisi.") from exc
        raise ApiError(f"L'API a répondu avec une erreur HTTP {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise ApiUnavailable(f"Impossible de joindre l'API ({API_URL}).") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ApiError("Réponse de l'API illisible (JSON invalide).") from exc


def api_is_healthy() -> bool:
    try:
        _request("GET", "/health")
        return True
    except ApiError:
        return False


def predict(text: str) -> str:
    data = _request("POST", "/predict", {"text": text})
    prediction = data.get("prediction") if isinstance(data, dict) else None
    if prediction not in VALID_PREDICTIONS:
        raise ApiError(f"Réponse inattendue de l'API : {data!r}")
    return prediction


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.set_page_config(page_title="SpamShield", page_icon="🛡️")

st.title("🛡️ SpamShield")
st.write("Collez un SMS ci-dessous : SpamShield indique s'il s'agit d'un **spam** ou d'un message **légitime**.")

with st.sidebar:
    st.subheader("État de l'API")
    if api_is_healthy():
        st.success("API disponible")
    else:
        st.error("API indisponible")
    st.caption(f"Adresse : {API_URL}")
    st.button("Actualiser")

sms = st.text_area("Texte du SMS", height=150, placeholder="Your SMS here")

if st.button("Analyser", type="primary"):
    if not sms.strip():
        st.warning("Veuillez saisir un SMS avant de lancer l'analyse.")
    else:
        try:
            with st.spinner("Analyse en cours..."):
                result = predict(sms)
        except ApiUnavailable as exc:
            st.error(f"🔌 API indisponible. {exc} Vérifiez que le service `api` est démarré.")
        except ApiError as exc:
            st.error(f"⚠️ Erreur API : {exc}")
        else:
            if result == "spam":
                st.error("🚫 **SPAM** — ce message est classé comme indésirable.")
            else:
                st.success("✅ **LÉGITIME** — ce message semble normal.")
