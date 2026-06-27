"""
Genera el diagrama Entidad-Relación de MyFraudLock con diseño moderno
usando Graphviz + HTML labels. Incluye iconos oficiales (PostgreSQL).

Convenciones visuales:
  - Cada entidad es una tabla con header coloreado y logo de PostgreSQL.
  - Badges PK (azul) y FK (naranja) junto a cada columna.
  - Cardinalidades indicadas en los labels de las aristas.
"""
import os
from graphviz import Digraph

BASE = os.path.dirname(__file__)
ICONS = os.path.join(BASE, "icons")
PG_ICON = os.path.join(ICONS, "postgresql.png")
OUT = os.path.join(BASE, "diagrama_bd")

PALETTE = {
    "usuario":       {"header": "#3182CE", "row": "#EBF8FF", "border": "#2C5282"},
    "transaccion":   {"header": "#805AD5", "row": "#FAF5FF", "border": "#553C9A"},
    "incidente":     {"header": "#E53E3E", "row": "#FFF5F5", "border": "#9B2C2C"},
    "configuracion": {"header": "#38A169", "row": "#F0FFF4", "border": "#22543D"},
}

PK_BADGE = '<TD BGCOLOR="#3182CE" WIDTH="34"><FONT COLOR="white" POINT-SIZE="9"><B>PK</B></FONT></TD>'
FK_BADGE = '<TD BGCOLOR="#DD6B20" WIDTH="34"><FONT COLOR="white" POINT-SIZE="9"><B>FK</B></FONT></TD>'
EMPTY_BADGE = '<TD WIDTH="34"></TD>'


def _row(name: str, dtype: str, badge: str = EMPTY_BADGE, note: str = "", bg: str = "#FFFFFF") -> str:
    note_html = (
        f'<TD ALIGN="LEFT"><FONT COLOR="#718096" POINT-SIZE="9"><I>{note}</I></FONT></TD>'
        if note else '<TD></TD>'
    )
    return (
        f'<TR>{badge}'
        f'<TD ALIGN="LEFT" BGCOLOR="{bg}"><FONT COLOR="#2D3748" POINT-SIZE="11"><B>{name}</B></FONT></TD>'
        f'<TD ALIGN="LEFT" BGCOLOR="{bg}"><FONT COLOR="#4A5568" POINT-SIZE="10">{dtype}</FONT></TD>'
        f'{note_html}'
        f'</TR>'
    )


def _table(entity: str, port_id: str, columns: list[tuple]) -> str:
    """columns: lista de tuplas (badge, name, dtype, note)"""
    colors = PALETTE[entity.lower()]
    header = (
        f'<TR><TD COLSPAN="4" BGCOLOR="{colors["header"]}" CELLPADDING="10">'
        f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
        f'<TR>'
        f'<TD FIXEDSIZE="TRUE" WIDTH="28" HEIGHT="28">'
        f'<IMG SRC="{PG_ICON}" SCALE="TRUE"/></TD>'
        f'<TD ALIGN="LEFT"><FONT COLOR="white" POINT-SIZE="16" FACE="Helvetica-Bold">'
        f'  {entity}</FONT></TD>'
        f'</TR>'
        f'</TABLE>'
        f'</TD></TR>'
    )
    rows = "".join(_row(n, t, b, note) for b, n, t, note in columns)
    return (
        f'<<TABLE BORDER="2" COLOR="{colors["border"]}" CELLBORDER="0" '
        f'CELLSPACING="0" CELLPADDING="6" PORT="{port_id}">'
        f'{header}{rows}</TABLE>>'
    )


g = Digraph(
    "MyFraudLock_BD",
    format="png",
    graph_attr={
        "label": "Modelo de Base de Datos  ·  MyFraudLock  (PostgreSQL)",
        "labelloc": "t",
        "fontname": "Helvetica-Bold",
        "fontsize": "22",
        "fontcolor": "#1A202C",
        "bgcolor": "#F7FAFC",
        "rankdir": "LR",
        "splines": "spline",
        "nodesep": "0.8",
        "ranksep": "1.4",
        "pad": "0.6",
    },
    node_attr={
        "shape": "plaintext",
        "fontname": "Helvetica",
    },
    edge_attr={
        "fontname": "Helvetica",
        "fontsize": "11",
        "color": "#4A5568",
        "penwidth": "1.8",
    },
)

# ─── Entidad Usuario ─────────────────────────────────────────────
g.node("Usuario", _table("Usuario", "usuario", [
    (PK_BADGE,    "id_usuario", "INTEGER",      "Identificador único"),
    (EMPTY_BADGE, "username",   "VARCHAR(100)", "único"),
    (EMPTY_BADGE, "email",      "VARCHAR(254)", "único"),
    (EMPTY_BADGE, "password",   "VARCHAR(128)", "hash"),
    (EMPTY_BADGE, "telefono",   "VARCHAR(15)",  "nullable"),
    (EMPTY_BADGE, "activo",     "BOOLEAN",      "default TRUE"),
    (EMPTY_BADGE, "rol",        "VARCHAR(15)",  "ADMIN · ANALISTA · GERENTE"),
]))

# ─── Entidad Transaccion ─────────────────────────────────────────
g.node("Transaccion", _table("Transaccion", "transaccion", [
    (PK_BADGE,    "id_transaccion",            "INTEGER",       "Identificador único"),
    (EMPTY_BADGE, "importe",                   "DECIMAL(12,2)", ""),
    (EMPTY_BADGE, "fecha",                     "TIMESTAMP",     "auto_now_add"),
    (EMPTY_BADGE, "card_brand",                "VARCHAR(20)",   "visa, master, amex..."),
    (EMPTY_BADGE, "card_type",                 "VARCHAR(10)",   "credit · debit"),
    (EMPTY_BADGE, "issuer_bank",               "VARCHAR(30)",   ""),
    (EMPTY_BADGE, "payment_channel",           "VARCHAR(15)",   "web · mobile · app"),
    (EMPTY_BADGE, "eci_code",                  "SMALLINT",      "3DS auth"),
    (EMPTY_BADGE, "num_installments",          "SMALLINT",      ""),
    (EMPTY_BADGE, "customer_region",           "VARCHAR(30)",   ""),
    (EMPTY_BADGE, "city_population",           "INTEGER",       ""),
    (EMPTY_BADGE, "is_new_customer",           "BOOLEAN",       ""),
    (EMPTY_BADGE, "days_since_first_purchase", "INTEGER",       ""),
    (EMPTY_BADGE, "avg_historical_amount",     "DECIMAL(12,2)", ""),
    (EMPTY_BADGE, "category",                  "VARCHAR(32)",   ""),
    (EMPTY_BADGE, "num_items",                 "SMALLINT",      ""),
    (EMPTY_BADGE, "has_discount",              "BOOLEAN",       ""),
    (EMPTY_BADGE, "previous_failed_attempts",  "SMALLINT",      ""),
    (EMPTY_BADGE, "session_duration_minutes",  "DECIMAL(6,2)",  "telemetría"),
    (EMPTY_BADGE, "interaction_velocity",      "DECIMAL(6,2)",  "telemetría"),
    (EMPTY_BADGE, "device_telemetry_1..5",     "DECIMAL(10,4)", "5 features"),
]))

# ─── Entidad Incidente ──────────────────────────────────────────
g.node("Incidente", _table("Incidente", "incidente", [
    (PK_BADGE,    "id_incidente",       "INTEGER",      "Identificador único"),
    (FK_BADGE,    "gestionado_por_id",  "INTEGER",      "→ Usuario"),
    (FK_BADGE,    "id_transaccion_id",  "INTEGER",      "→ Transaccion (1:1)"),
    (EMPTY_BADGE, "comentario",         "TEXT",         "nullable"),
    (EMPTY_BADGE, "estado",             "VARCHAR(20)",  "Pendiente · Fraude · Falso+"),
    (EMPTY_BADGE, "es_fraude",          "BOOLEAN",      "default FALSE"),
    (EMPTY_BADGE, "fecha",              "TIMESTAMP",    "auto_now_add"),
    (EMPTY_BADGE, "score_riesgo",       "DECIMAL(5,2)", "% de riesgo"),
    (EMPTY_BADGE, "explicabilidad",     "JSONB",        "SHAP values"),
]))

# ─── Entidad Configuracion ──────────────────────────────────────
g.node("Configuracion", _table("Configuracion", "configuracion", [
    (PK_BADGE,    "id",                   "INTEGER",   "Identificador único"),
    (EMPTY_BADGE, "umbral_score",         "INTEGER",   "default 70"),
    (EMPTY_BADGE, "notificaciones_email", "BOOLEAN",   "default TRUE"),
    (EMPTY_BADGE, "actualizado_en",       "TIMESTAMP", "auto_now"),
    (FK_BADGE,    "actualizado_por_id",   "INTEGER",   "→ Usuario"),
]))

# ─── Relaciones ────────────────────────────────────────────────
g.edge(
    "Usuario", "Incidente",
    label="  gestiona  ", taillabel="1", headlabel="0..N",
    arrowhead="crow", arrowtail="teetee", dir="both",
    color=PALETTE["incidente"]["header"],
    labelfontcolor="#2D3748", labeldistance="2", labelangle="20",
)
g.edge(
    "Transaccion", "Incidente",
    label="  genera  ", taillabel="1", headlabel="0..1",
    arrowhead="teeodot", arrowtail="teetee", dir="both",
    color=PALETTE["transaccion"]["header"],
    labelfontcolor="#2D3748", labeldistance="2", labelangle="-20",
)
g.edge(
    "Usuario", "Configuracion",
    label="  actualiza  ", taillabel="1", headlabel="0..N",
    arrowhead="crow", arrowtail="teetee", dir="both",
    color=PALETTE["configuracion"]["header"],
    labelfontcolor="#2D3748", labeldistance="2", labelangle="-20",
)

g.render(OUT, cleanup=True)
print(f"Diagrama generado: {OUT}.png")
