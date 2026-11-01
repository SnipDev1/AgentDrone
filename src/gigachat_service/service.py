"""GigaChat API integration service for Siberian Bank collateral monitoring.

Provides an AI-powered expert inspector summary using Sber GigaChat SDK
with a deterministic, professional offline fallback generator.
Supports both banking domain models and CV inspection pipeline schemas.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import ssl
import time
from typing import Any, Optional
import urllib.parse
import urllib.request
import uuid

from PIL import Image, ImageDraw

from .prompts import SYSTEM_INSPECTOR_PROMPT, build_inspector_prompt, extract_context_payload

logger = logging.getLogger(__name__)

_VISION_AGENT_CACHE: dict[str, dict[str, Any]] = {}


def _read_env_credentials() -> Optional[str]:
    """Read GIGACHAT_CREDENTIALS from environment or project .env file."""
    val = os.getenv("GIGACHAT_CREDENTIALS")
    if val:
        return val.strip()
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GIGACHAT_CREDENTIALS="):
                    _, v = line.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    if v:
                        os.environ["GIGACHAT_CREDENTIALS"] = v
                        return v
        except Exception:
            pass
    return None


class _RestChoiceMessage:
    def __init__(self, content: str):
        self.content = content


class _RestChoice:
    def __init__(self, content: str):
        self.message = _RestChoiceMessage(content)


class _RestChatResponse:
    def __init__(self, content: str):
        self.choices = [_RestChoice(content)]


class _SberGigaChatRestClient:
    """Direct HTTPS REST client for Sber GigaChat API (zero external dependencies)."""

    OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    FILES_URL = "https://gigachat.devices.sberbank.ru/api/v1/files"

    def __init__(self, credentials: str, scope: str = "GIGACHAT_API_PERS", model: str = "GigaChat"):
        self.credentials = credentials.strip()
        self.scope = scope
        self.model = model
        self._access_token: Optional[str] = None
        self._expires_at: float = 0.0
        self._ssl_ctx = ssl._create_unverified_context()

    def _get_token(self) -> str:
        now = time.time()
        if self._access_token and now < self._expires_at - 60:
            return self._access_token

        req = urllib.request.Request(
            self.OAUTH_URL,
            data=urllib.parse.urlencode({"scope": self.scope}).encode("utf-8"),
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {self.credentials}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=8.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self._access_token = data["access_token"]
            exp_ms = data.get("expires_at", (now + 1700) * 1000)
            self._expires_at = float(exp_ms) / 1000.0 if exp_ms > 1e11 else float(exp_ms)
            return self._access_token

    def upload_image(self, jpeg_bytes: bytes, filename: str = "collage.jpg") -> str:
        """Upload an image to Sber GigaChat Files API and return file_id."""
        token = self._get_token()
        boundary = "----GigaChatBoundary" + uuid.uuid4().hex
        body = (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f'Content-Type: image/jpeg\r\n\r\n'
        ).encode("utf-8") + jpeg_bytes + (
            f'\r\n--{boundary}\r\n'
            f'Content-Disposition: form-data; name="purpose"\r\n\r\ngeneral\r\n'
            f'--{boundary}--\r\n'
        ).encode("utf-8")

        req = urllib.request.Request(
            self.FILES_URL,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=15.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return str(data["id"])

    def chat(self, payload: dict[str, Any]) -> _RestChatResponse:
        token = self._get_token()
        req_model = payload.get("model") or self.model
        has_attachments = any(
            bool(m.get("attachments")) for m in payload.get("messages", []) if isinstance(m, dict)
        )
        candidate_models = [req_model]
        fallback_list = (
            ("GigaChat-2-Max", "GigaChat-Max")
            if has_attachments
            else ("GigaChat-2-Max", "GigaChat-2-Pro", "GigaChat")
        )
        for fallback_mdl in fallback_list:
            if fallback_mdl not in candidate_models:
                candidate_models.append(fallback_mdl)

        last_err: Optional[Exception] = None
        for mdl in candidate_models:
            body_dict = dict(payload)
            body_dict["model"] = mdl
            req = urllib.request.Request(
                self.CHAT_URL,
                data=json.dumps(body_dict).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=30.0) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    content = res["choices"][0]["message"]["content"]
                    return _RestChatResponse(content)
            except Exception as exc:
                last_err = exc
                continue
        raise RuntimeError(f"GigaChat REST call failed: {last_err}")


class GigaChatInspectorService:
    """Generates official inspection summaries via GigaChat API with offline fallback."""

    def __init__(
        self,
        credentials: Optional[str] = None,
        scope: Optional[str] = None,
        model: str = "GigaChat",
        verify_ssl_certs: bool = False,
    ):
        if credentials is not None:
            self.credentials = credentials
        else:
            self.credentials = _read_env_credentials()
        self.scope = scope or os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.model = model or os.getenv("GIGACHAT_MODEL", "GigaChat")
        self.verify_ssl_certs = verify_ssl_certs
        self._client: Any = None
        self._is_online = False
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Sber GigaChat REST API client (`gigachat.devices.sberbank.ru`)."""
        if not self.credentials:
            logger.info("GigaChat credentials not provided. Operating in offline template fallback mode.")
            self._is_online = False
            self._client = None
            return

        try:
            self._client = _SberGigaChatRestClient(
                credentials=self.credentials,
                scope=self.scope,
                model=self.model if self.model != "GigaChat-2-Pro" else "GigaChat",
            )
            self._is_online = True
            logger.info("Sber GigaChat REST API client initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize GigaChat client: {e}. Falling back to offline mode.")
            self._client = None
            self._is_online = False

    def set_credentials(
        self,
        credentials: str,
        scope: Optional[str] = None,
        model: Optional[str] = None,
    ) -> bool:
        """Re-initialize client with runtime credentials (e.g. from Streamlit UI)."""
        self.credentials = credentials.strip()
        if scope:
            self.scope = scope
        if model:
            self.model = model
        self._init_client()
        return self.is_online

    @property
    def is_online(self) -> bool:
        """Check if active GigaChat client is available."""
        return self._is_online and self._client is not None

    def get_mode_status(self) -> dict[str, Any]:
        """Return human-readable service status for UI indicators."""
        return {
            "is_online": self.is_online,
            "mode": "Онлайн (GigaChat API Сбер)" if self.is_online else "Оффлайн (Шаблонизатор Сбера)",
            "model": self.model if self.is_online else "Offline-Fallback-Engine",
            "scope": self.scope,
            "has_credentials": bool(self.credentials),
        }

    def generate_summary(
        self,
        credit: Any,
        inspection: Any,
        recommendation: Any,
    ) -> str:
        """Generate expert inspector summary (via API or offline fallback)."""
        payload = extract_context_payload(credit, inspection, recommendation)

        if self.is_online and self._client:
            try:
                user_content = build_inspector_prompt(credit, inspection, recommendation)
                response = self._client.chat({
                    "messages": [
                        {"role": "system", "content": SYSTEM_INSPECTOR_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 600,
                })
                if response and response.choices and len(response.choices) > 0:
                    text = response.choices[0].message.content.strip()
                    if text:
                        return text
            except Exception as e:
                if "No such model" in str(e) and self.model != "GigaChat":
                    try:
                        self.model = "GigaChat"
                        self._init_client()
                        if self._client:
                            response = self._client.chat({
                                "messages": [
                                    {"role": "system", "content": SYSTEM_INSPECTOR_PROMPT},
                                    {"role": "user", "content": user_content},
                                ],
                                "temperature": 0.3,
                                "max_tokens": 600,
                            })
                            if response and response.choices and len(response.choices) > 0:
                                text = response.choices[0].message.content.strip()
                                if text:
                                    return text
                    except Exception as retry_err:
                        e = retry_err
                logger.warning(f"GigaChat API call failed: {e}. Generating offline expert summary.")

        # Offline fallback
        return self._generate_offline_summary(payload)

    def analyze_plot_with_vision_agent(
        self,
        cv_result: Any,
        credit_obj: Any = None,
        parcel: Any = None,
        plot_dir: Optional[str | Path] = None,
    ) -> Optional[dict[str, Any]]:
        """Multimodal Sber AI Agent (GigaChat-2-Max Vision): inspects drone photos + RTK telemetry + cadastre."""
        if not self.is_online or not isinstance(self._client, _SberGigaChatRestClient) or not plot_dir:
            return None

        p_dir = Path(plot_dir)
        if not p_dir.exists():
            return None

        jpgs = sorted([f for f in p_dir.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg")])
        if not jpgs:
            return None

        cache_key = hashlib.md5(
            f"{p_dir.resolve()}:{len(jpgs)}:{getattr(cv_result, 'cadastral_id', '')}".encode("utf-8")
        ).hexdigest()

        if cache_key not in _VISION_AGENT_CACHE:
            try:
                shots_list = getattr(cv_result, "shots_metadata", None) or getattr(cv_result, "photos_meta", [])
                meta_by_name = {
                    getattr(m, "filename", ""): m for m in shots_list
                }

                # Smart frame selection: prioritize perspective/facade shots (pitch -25..-65 deg)
                # showing the building walls from different sides, plus 1 nadir shot for roof & layout
                facade_shots = []
                nadir_shots = []
                for p in jpgs:
                    m = meta_by_name.get(p.name)
                    pitch = getattr(m, "gimbal_pitch_deg", getattr(m, "gimbal_pitch", -90.0)) if m else -90.0
                    if -65.0 <= pitch <= -25.0:
                        facade_shots.append(p)
                    else:
                        nadir_shots.append(p)

                selected_files: list[Path] = []
                if facade_shots:
                    # Pick up to 3 diverse facade shots
                    if len(facade_shots) <= 3:
                        selected_files.extend(facade_shots)
                    else:
                        step = len(facade_shots) / 3.0
                        selected_files.extend([facade_shots[int(i * step)] for i in range(3)])
                    # Add 1 nadir shot
                    if nadir_shots:
                        selected_files.append(nadir_shots[0])
                    elif len(facade_shots) > 3:
                        selected_files.append(facade_shots[-1])
                else:
                    # Fallback to evenly spaced shots across the flight
                    n = len(jpgs)
                    if n >= 4:
                        indices = [0, max(1, n // 4), max(2, n // 2), n - 1]
                    else:
                        indices = list(range(n))
                    selected_files = [jpgs[i] for i in indices[:4]]

                # Ensure exactly up to 4 distinct files
                seen_paths: set[Path] = set()
                final_selected: list[Path] = []
                for f in selected_files:
                    if f not in seen_paths and len(final_selected) < 4:
                        seen_paths.add(f)
                        final_selected.append(f)
                for f in jpgs:
                    if f not in seen_paths and len(final_selected) < 4:
                        seen_paths.add(f)
                        final_selected.append(f)
                selected_files = final_selected

                c_lat = getattr(parcel, "centroid_lat", None)
                c_lon = getattr(parcel, "centroid_lon", None)
                if c_lat is None and shots_list:
                    lats = [getattr(m, "latitude", getattr(m, "lat", 0.0)) for m in shots_list]
                    lons = [getattr(m, "longitude", getattr(m, "lon", 0.0)) for m in shots_list]
                    lats = [v for v in lats if v != 0]
                    lons = [v for v in lons if v != 0]
                    if lats and lons:
                        c_lat = sum(lats) / len(lats)
                        c_lon = sum(lons) / len(lons)

                # High-definition collage (960x720 per quadrant = 1920x1440 total)
                W, H = 960, 720
                collage = Image.new("RGB", (W * 2, H * 2), (24, 28, 36))
                draw = ImageDraw.Draw(collage)
                frame_descriptions: list[str] = []

                for idx, img_path in enumerate(selected_files):
                    im = Image.open(img_path).convert("RGB").resize((W, H))
                    x = (idx % 2) * W
                    y = (idx // 2) * H
                    collage.paste(im, (x, y))

                    m = meta_by_name.get(img_path.name)
                    alt = getattr(m, "relative_altitude_m", getattr(m, "rel_alt", 0.0)) if m else 0.0
                    pitch = getattr(m, "gimbal_pitch_deg", getattr(m, "gimbal_pitch", -90.0)) if m else -90.0
                    m_lat = getattr(m, "latitude", getattr(m, "lat", 0.0)) if m else 0.0
                    m_lon = getattr(m, "longitude", getattr(m, "lon", 0.0)) if m else 0.0
                    dist_m = 0.0
                    if m and c_lat and c_lon and m_lat != 0:
                        dlat = (m_lat - c_lat) * 111320.0
                        dlon = (m_lon - c_lon) * 111320.0 * math.cos(math.radians(c_lat))
                        dist_m = math.hypot(dlat, dlon)

                    draw.rectangle([x, y, x + W, y + 28], fill=(15, 23, 42))
                    draw.text(
                        (x + 10, y + 6),
                        f"Frame {idx + 1} ({img_path.name}) | Alt:{alt:.1f}m | Pitch:{pitch:.0f}deg | Dist:{dist_m:.0f}m",
                        fill=(255, 255, 255),
                    )
                    frame_descriptions.append(
                        f"- Кадр {idx + 1} ({img_path.name}): высота {alt:.1f} м, угол наклона камеры {pitch:.1f}°, "
                        f"удаление точки съемки от центра кадастрового участка: {dist_m:.1f} м"
                    )

                buf = io.BytesIO()
                collage.save(buf, format="JPEG", quality=88)
                file_id = self._client.upload_image(buf.getvalue(), filename=f"inspection_{p_dir.name}.jpg")

                cadastral_str = (
                    getattr(parcel, "cadastral_id", None)
                    or getattr(credit_obj, "cadastral_id", None)
                    or getattr(cv_result, "cadastral_id", "")
                )
                address_str = (
                    getattr(credit_obj, "address", None)
                    or getattr(parcel, "address", None)
                    or "Н/Д"
                )
                vri_str = getattr(parcel, "permitted_use", None) or "ИЖС"
                area_val = getattr(parcel, "area_sqm", None) or 800.0
                start_date_str = getattr(credit_obj, "construction_start_date", "Нет")
                plan_date_str = getattr(credit_obj, "planned_completion_date", "Нет")
                pz_str = getattr(credit_obj, "has_pz", "Нет")

                cv_mat_val = getattr(cv_result, "detected_material", None)
                cv_mat_str = cv_mat_val.value if hasattr(cv_mat_val, "value") else str(cv_mat_val or "")
                cv_conf = getattr(cv_result, "material_confidence", 0.0)

                vision_prompt = (
                    "Ты — экспертный мультимодальный ИИ-агент строительного мониторинга залогов ИЖС ПАО Сбербанк.\n"
                    f"Кадастровый номер участка: {cadastral_str}. Территория: {address_str}.\n"
                    f"ВРИ: {vri_str}. Площадь участка: {area_val:.0f} кв. м.\n"
                    f"Параметры договора: дата начала: {start_date_str}, план завершения: {plan_date_str}, наличие ПЗ: {pz_str}.\n\n"
                    "На изображении представлен коллаж из 4 снимков БПЛА высокой четкости:\n"
                    + "\n".join(frame_descriptions)
                    + "\n\nВНИМАНИЕ: Оценивай строго ГЛАВНЫЙ жилой дом (капитальное жилое здание на участке). "
                    "Игнорируй временные сараи, хозблоки, навесы, теплицы или стройматериалы во дворе.\n\n"
                    f"Данные спектрального компьютерного зрения (CV): доминирующий оптический спектр стен указывает на '{cv_mat_str}' (уверенность {cv_conf:.0%}).\n\n"
                    "Внимательно изучи снимки и проведи объективную оценку строения строго по видимым фактам:\n"
                    "1. МАТЕРИАЛ НЕСУЩИХ СТЕН ГЛАВНОГО ДОМА:\n"
                    "   - Кирпич: стены из красного, рыжего или терракотового кирпича (кирпичная кладка, облицовочный кирпич, керамический блок);\n"
                    "   - Газобетон/Блок: светло-серые или белые прямоугольные блоки со швами (газобетон, пеноблок, сибит, шлакоблок);\n"
                    "   - Брус/Дерево: бревенчатые или брусовые деревянные венцы;\n"
                    "   - Каркас/Сайдинг: щитовой каркас, сайдинг, фасадные панели;\n"
                    "   - Сэндвич-панели/Коммерческий: крупные металлические кассеты, сэндвич-панели, профлист.\n"
                    "2. КРОВЛЯ: укажи тип и цвет покрытия крыши (металлочерепица, профлист, шифер, мягкая кровля или отсутствует).\n"
                    "3. СТАДИЯ И ГОТОВНОСТЬ:\n"
                    "   - 0%: участок пустой, котлован или только фундамент;\n"
                    "   - 50%: возведена коробка несущих стен (под крышей или без), завершен основной конструктив;\n"
                    "   - 100%: готовый дом с закрытым тепловым контуром (окна, двери, кровля);\n"
                    "   - процент готовности от 0 до 100%.\n"
                    "4. ЦЕЛЕВОЕ НАЗНАЧЕНИЕ: определи, является ли объект индивидуальным жилым домом (ИЖС) или на снимках видно коммерческое/торговое/складское здание (вывески 'Магазин'/'Супермаркет', рампа, торговый зал, автобаза). Если это коммерческий объект — установи `is_commercial_building: true`, `fraud_risk_flag: true` и укажи причину в `fraud_reasons`.\n"
                    "5. ДЕФЕКТЫ: укажи только реально видимые строительные дефекты или реальные помехи съемке (если нет — оставь []).\n\n"
                    "Выведи JSON строго в формате:\n"
                    "```json\n"
                    "{\n"
                    '  "stage": "0%" | "50%" | "100%",\n'
                    '  "readiness_pct": число_от_0_до_100,\n'
                    '  "detected_material": "Газобетон/Блок" | "Кирпич" | "Брус/Дерево" | "Каркас/Сайдинг" | "Сэндвич-панели/Коммерческий",\n'
                    '  "roof_material": "описание материала и цвета кровли",\n'
                    '  "is_commercial_building": true | false,\n'
                    '  "fraud_risk_flag": true | false,\n'
                    '  "fraud_reasons": ["строка"],\n'
                    '  "defects_detected": ["строка"],\n'
                    '  "expert_summary": "Объективное экспертное заключение (2-4 предложения) строго по видимым на фото фактам"\n'
                    "}\n"
                    "```"
                )

                resp = self._client.chat({
                    "model": "GigaChat-2-Max",
                    "messages": [
                        {
                            "role": "user",
                            "content": vision_prompt,
                            "attachments": [file_id],
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 900,
                })
                raw_text = resp.choices[0].message.content.strip()
                code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
                json_match = code_match or re.search(r"(\{[^{}]*\"expert_summary\"[^{}]*\})", raw_text, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(1))
                    except Exception:
                        parsed = {"expert_summary": raw_text}
                else:
                    parsed = {"expert_summary": raw_text}
                _VISION_AGENT_CACHE[cache_key] = parsed
            except Exception as exc:
                logger.warning(f"Multimodal GigaChat-2-Max vision analysis failed: {exc}")
                return None

        agent_data = _VISION_AGENT_CACHE.get(cache_key)
        if not agent_data:
            return None

        # Synchronize detected wall material from Vision AI Agent with cv_result
        v_mat = (
            agent_data.get("detected_material")
            or agent_data.get("материал_стен")
            or agent_data.get("material")
        )
        if v_mat:
            v_mat_lower = str(v_mat).lower()
            from ..models.schemas import WallMaterial as SchemaWallMaterial
            if "кирпич" in v_mat_lower:
                cv_result.detected_material = SchemaWallMaterial.BRICK
                cv_result.material_confidence = max(getattr(cv_result, "material_confidence", 0.0), 0.95)
            elif any(k in v_mat_lower for k in ("газобетон", "пеноблок", "блок", "сибит", "шлакоблок")):
                # Guard against hallucination if CV has strong optical evidence for brick
                cv_is_brick = (
                    getattr(cv_result, "detected_material", None) == SchemaWallMaterial.BRICK
                    and getattr(cv_result, "material_confidence", 0.0) >= 0.85
                )
                if cv_is_brick:
                    logger.info("Retaining high-confidence CV Brick detection over vision agent block label")
                else:
                    cv_result.detected_material = SchemaWallMaterial.GAS_CONCRETE
                    cv_result.material_confidence = max(getattr(cv_result, "material_confidence", 0.0), 0.95)
            elif any(k in v_mat_lower for k in ("дерев", "брус", "бревн")):
                cv_result.detected_material = SchemaWallMaterial.TIMBER
                cv_result.material_confidence = max(getattr(cv_result, "material_confidence", 0.0), 0.95)
            elif any(k in v_mat_lower for k in ("каркас", "щит", "сайдинг", "sip")):
                cv_result.detected_material = SchemaWallMaterial.FRAME_PANEL
                cv_result.material_confidence = max(getattr(cv_result, "material_confidence", 0.0), 0.95)
            elif any(k in v_mat_lower for k in ("сэндвич", "коммерческ")):
                cv_result.detected_material = SchemaWallMaterial.SANDWICH_PANEL
                cv_result.material_confidence = max(getattr(cv_result, "material_confidence", 0.0), 0.95)

        # Synchronize commercial / fraud findings
        if agent_data.get("is_commercial_building") or agent_data.get("fraud_risk_flag"):
            cv_result.fraud_risk_flag = True
            for r in agent_data.get("fraud_reasons") or []:
                r_str = str(r).strip()
                if r_str and r_str not in cv_result.fraud_reasons:
                    cv_result.fraud_reasons.append(r_str)

        for defect in agent_data.get("defects_detected") or []:
            defect_str = str(defect).strip()
            if defect_str and defect_str not in cv_result.defects_detected:
                cv_result.defects_detected.append(defect_str)

        summary = str(agent_data.get("expert_summary") or "").strip()
        from ..models.schemas import WallMaterial as SchemaWallMaterial
        if cv_result.detected_material == SchemaWallMaterial.BRICK and summary and "газобетон" in summary.lower():
            summary = summary.replace("газобетонных блоков", "кирпича").replace("газобетона", "кирпича")
        if summary:
            if isinstance(getattr(cv_result, "operator_notes", None), list):
                if summary not in cv_result.operator_notes:
                    cv_result.operator_notes.append(summary)
            else:
                cv_result.operator_notes = [summary]

        return agent_data

    def generate_summary_for_ui(
        self,
        cv_result: Any,
        credit_obj: Any,
        decision: Any,
        plot_dir: Optional[str | Path] = None,
        parcel: Any = None,
    ) -> str:
        """Convenience method for Streamlit UI passing CV pipeline schemas."""
        if plot_dir and self.is_online:
            agent_data = self.analyze_plot_with_vision_agent(
                cv_result=cv_result,
                credit_obj=credit_obj,
                parcel=parcel,
                plot_dir=plot_dir,
            )
            if agent_data and agent_data.get("expert_summary"):
                return str(agent_data["expert_summary"]).strip()
        op_notes = getattr(cv_result, "operator_notes", None)
        if op_notes and self.is_online:
            if isinstance(op_notes, list):
                return "\n\n".join(str(x) for x in op_notes if x).strip()
            return str(op_notes).strip()
        return self.generate_summary(credit=credit_obj, inspection=cv_result, recommendation=decision)

    def _generate_offline_summary(self, payload: dict[str, Any]) -> str:
        """Generate a deterministic, legally accurate bank summary for AS 'Kalita'."""
        credit_info = payload.get("кредитный_договор", {})
        insp_info = payload.get("результаты_бпла_осмотра", {})
        dec_info = payload.get("решение_decision_engine", {})

        cadastral = credit_info.get("кадастровый_номер", "")
        borrower = credit_info.get("заемщик", "Заемщик")
        overdue = credit_info.get("просрочка_дней", 0)
        proj_mat = credit_info.get("проектный_материал", "")
        deadline = credit_info.get("дедлайн_стройки", "")

        date_str = insp_info.get("дата_съемки", "")
        readiness_pct = float(insp_info.get("физическая_готовность_процент", 0.0))
        det_mat = insp_info.get("фактический_материал", "")
        is_commercial = insp_info.get("риск_коммерческого_использования", False)
        mat_match = insp_info.get("соответствие_материала", True)

        strat_str = str(dec_info.get("стратегия", ""))
        strat_code = str(dec_info.get("код_решения", ""))

        header = f"ЗАКЛЮЧЕНИЕ ИНСПЕКТОРА ПО ОБЪЕКТУ {cadastral}:\n"

        # 1. STOP-FRAUD
        if "FRAUD" in strat_code or "Блокировка" in strat_str or is_commercial or not mat_match:
            if is_commercial:
                return (
                    f"{header}"
                    f"В ходе аэромониторинга БПЛА от {date_str} на участке ИЖС "
                    "зафиксировано возведение нежилого коммерческого объекта (склад / супермаркет / автосервис). "
                    "Выявлено прямое нарушение целевого назначения земельного участка. "
                    "В соответствии со ст. 813, 814 ГК РФ выдача траншей заблокирована, кредитное досье передано "
                    "в Управление безопасности Сибирского банка для проведения служебного расследования."
                )
            else:
                return (
                    f"{header}"
                    f"По результатам обследования БПЛА выявлено критическое расхождение фактического материала несущих стен "
                    f"({det_mat}) с утвержденной проектно-сметной документацией ({proj_mat}). "
                    "Имеются признаки умышленного удешевления сметной стоимости предмета залога. "
                    "Кредитование временно приостановлено. Назначена строительно-техническая экспертиза сметы."
                )

        # 2. DEFAULT & FORECLOSURE
        if "DEFAULT" in strat_code or "LEGAL" in strat_code or "Дефолт" in strat_str or "Взыскание" in strat_str:
            return (
                f"{header}"
                f"Объект находится в зоне дефолтного риска. При нормативном сроке ввода до {deadline} "
                f"фактическая строительная готовность строения составляет {readiness_pct:.1f}% (просрочка {overdue} дн.). "
                "Физическая возможность завершения строительства в договорные сроки отсутствует. "
                "Рекомендован запуск процедуры досрочного истребования задолженности и судебного обращения взыскания "
                "на обеспечение для последующей реализации на торгах."
            )

        # 3. PLEDGE REGISTRATION
        if "REGISTER" in strat_code or "PLEDGE" in strat_code or "Регистрация" in strat_str:
            return (
                f"{header}"
                f"Строительство жилого дома успешно завершено (фактическая готовность {readiness_pct:.1f}%, "
                "тепловой контур полностью закрыт). Объект готов к оформлению в качестве постоянного обеспечения Банка. "
                f"Заемщику {borrower} направлено официальное уведомление о необходимости подачи документов "
                "на регистрацию права собственности и залога жилого дома в пользу Сибирского банка ПАО Сбербанк."
            )

        # 4. RESTRUCTURING
        if "RESTRUCTURE" in strat_code or "Реструктуризация" in strat_str or overdue > 0:
            return (
                f"{header}"
                f"Зафиксировано отставание от проектного графика строительства (фактическая готовность {readiness_pct:.1f}%, "
                f"просрочка платежей {overdue} дн., срок завершения: {deadline}). "
                "В целях недопущения перехода кредита в категорию NPL и защиты финансового положения заемщика "
                "рекомендована программа реструктуризации: предоставление кредитных каникул и пролонгация нормативного срока "
                "строительства с утверждением скорректированного графика траншей."
            )

        # 5. TRANCHE APPROVAL
        return (
            f"{header}"
            "Строительство объекта ведется в строгом соответствии с графиком кредитного соглашения и проектной сметой. "
            f"Фактическая готовность {readiness_pct:.1f}% подтверждает выполнение контрольного этапа. Несущие стены "
            f"выполнены из проектного материала ({proj_mat}), градостроительные отступы соблюдены, просрочка отсутствует. "
            "Автоматически согласован очередной транш в режиме сквозной обработки STP без созыва кредитного комитета."
        )

    def analyze_construction_dynamics(
        self,
        dynamics: Any,
        credit: Any = None,
    ) -> str:
        """Generate official AI Agent inspector summary on multi-temporal construction dynamics."""
        from .prompts import SYSTEM_DYNAMICS_PROMPT, build_dynamics_prompt

        if self.is_online and self._client:
            try:
                user_content = build_dynamics_prompt(dynamics, credit)
                response = self._client.chat({
                    "messages": [
                        {"role": "system", "content": SYSTEM_DYNAMICS_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 700,
                })
                if response and response.choices and len(response.choices) > 0:
                    text = response.choices[0].message.content.strip()
                    if text:
                        return text
            except Exception as e:
                logger.warning(f"GigaChat dynamics API call failed: {e}. Generating offline dynamics summary.")

        return self._generate_offline_dynamics_summary(dynamics, credit)

    def _generate_offline_dynamics_summary(self, dynamics: Any, credit: Any = None) -> str:
        """Deterministic, domain-accurate banking dynamics summary for AS 'Kalita'."""
        cadastral = getattr(dynamics, "cadastral_id", "54:00:000000:0")
        delta_pct = getattr(dynamics, "delta_readiness_pct", 0.0)
        delta_days = getattr(dynamics, "delta_days", 0)
        pace = getattr(dynamics, "pace_pct_per_month", 0.0)
        pace_status = getattr(dynamics, "pace_status", "В графике")
        elements = getattr(dynamics, "elements_progress", [])
        snapshots = getattr(dynamics, "snapshots", [])
        planned_date = getattr(credit, "planned_completion_date", "Не указан") if credit else "Не указан"

        d1 = snapshots[0].flight_date if snapshots else "Дата 1"
        r1 = snapshots[0].readiness_pct if snapshots else 0.0
        d2 = snapshots[-1].flight_date if len(snapshots) > 1 else d1
        r2 = snapshots[-1].readiness_pct if len(snapshots) > 1 else r1

        elem_text = "; ".join(elements) if elements else f"Прирост готовности составил +{delta_pct}%"

        header = f"ЭКСПЕРТНОЕ ЗАКЛЮЧЕНИЕ ИИ-ИНСПЕКТОРА ПО ДИНАМИКЕ СТРОИТЕЛЬСТВА ({cadastral}):\n"

        if len(snapshots) < 2:
            single = snapshots[0] if snapshots else None
            r_str = f"{single.readiness_pct:.1f}%" if single else "0.0%"
            st_title = single.stage_title if single else "Стадия 0%"
            return (
                f"{header}"
                f"По объекту проведена первичная аэрофотосъемка с БПЛА от {d1}. "
                f"Фактическая строительная готовность зафиксирована на уровне {r_str} ({st_title}). "
                "Сравнительный расчет динамики и темпов СМР (%/мес) будет автоматически сформирован после выполнения "
                "повторного контрольного облета БПЛА.\n\n"
                f"С учетом планового срока ввода ({planned_date}) текущий статус принят к мониторингу. "
                "Рекомендация для залоговой службы: зафиксировать контрольную веху и запланировать повторный облет БПЛА."
            )

        if delta_pct <= 1.0 and delta_days >= 30:
            return (
                f"{header}"
                f"По результатам сравнительного анализа серий аэрофотосъемки БПЛА от {d1} и {d2} "
                f"(интервал {delta_days} дн.) на участке зафиксирована стагнация СМР: "
                f"фактический прирост готовности составил {delta_pct:+.1f}% (готовность остается на уровне {r2:.1f}%). "
                "Признаки ведения активных строительно-монтажных работ отсутствуют. "
                f"С учетом планового срока ввода ({planned_date}) формируется критический риск долгостроя и нарушения условий кредитования.\n\n"
                "Решение для залоговой службы: приостановить согласование траншей, направить заемщику требование "
                "о предоставлении отчета о расходовании ранее выданных средств и инициировать выездную инспекцию залога."
            )

        if pace >= 15.0 or delta_pct >= 50.0:
            return (
                f"{header}"
                f"Сравнительный анализ данных БПЛА за период с {d1} по {d2} (интервал {delta_days} дн.) "
                f"демонстрирует высокую динамику возведения объекта: прирост физической готовности составил +{delta_pct:.1f}% "
                f"(с {r1:.1f}% до {r2:.1f}%). "
                f"Зафиксированы ключевые конструктивные изменения: {elem_text}.\n\n"
                f"Расчетный темп СМР составляет {pace:.1f}% в месяц ({pace_status}), что полностью соответствует графику "
                f"финансирования с запасом по плановому сроку ввода ({planned_date}). "
                "Целевое расходование ранее предоставленного кредитного транша подтверждено фотограмметрией. "
                "Рекомендация для залоговой службы: согласовать выдачу очередного транша в режиме «Зеленый коридор» (STP)."
            )

        return (
            f"{header}"
            f"В результате сопоставления разновременных снимков БПЛА от {d1} и {d2} (прошло {delta_days} дн.) "
            f"установлена положительная динамика СМР: готовность объекта возросла на +{delta_pct:.1f}% "
            f"(с {r1:.1f}% до {r2:.1f}%). Выполнены следующие работы: {elem_text}.\n\n"
            f"Среднемесячный темп строительства зафиксирован на уровне {pace:.1f}%/мес ({pace_status}). "
            f"Объект возводится в пределах допустимых календарных отклонений относительно планового срока ({planned_date}). "
            "Рекомендация для залоговой службы: подтвердить целевой статус этапа и продолжить плановый кредитный мониторинг."
        )
