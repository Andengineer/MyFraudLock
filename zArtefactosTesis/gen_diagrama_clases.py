"""
Genera el diagrama de Clases UML de MyFraudLock con diseño moderno
usando Graphviz + HTML labels e iconos oficiales (Django, Python, PostgreSQL, ML).
"""
import os
from graphviz import Digraph

BASE = os.path.dirname(__file__)
ICONS = os.path.join(BASE, "icons")
OUT = os.path.join(BASE, "diagrama_clases")

ICON = {
    "django":  os.path.join(ICONS, "django.png"),
    "python":  os.path.join(ICONS, "python.png"),
    "js":      os.path.join(ICONS, "javascript.png"),
    "ml":      os.path.join(ICONS, "ml.png"),
    "pg":      os.path.join(ICONS, "postgresql.png"),
}

# Paleta por estereotipo (capa / responsabilidad)
PALETTE = {
    "controller": {"header": "#2B6CB0", "body": "#EBF8FF", "border": "#1A365D"},
    "service":    {"header": "#805AD5", "body": "#FAF5FF", "border": "#553C9A"},
    "ml":         {"header": "#E53E3E", "body": "#FFF5F5", "border": "#9B2C2C"},
    "data":       {"header": "#38A169", "body": "#F0FFF4", "border": "#22543D"},
    "view":       {"header": "#D69E2E", "body": "#FFFFF0", "border": "#744210"},
    "frontend":   {"header": "#319795", "body": "#E6FFFA", "border": "#234E52"},
}


def _section(title: str, items: list[str], bg: str) -> str:
    if not items:
        return (
            f'<TR><TD BGCOLOR="{bg}" ALIGN="LEFT" CELLPADDING="6">'
            f'<FONT COLOR="#A0AEC0" POINT-SIZE="9"><I>· {title} ·</I></FONT>'
            f'</TD></TR>'
        )
    header = (
        f'<TR><TD BGCOLOR="{bg}" ALIGN="LEFT" CELLPADDING="4">'
        f'<FONT COLOR="#718096" POINT-SIZE="9"><B>« {title} »</B></FONT></TD></TR>'
    )
    rows = "".join(
        f'<TR><TD BGCOLOR="{bg}" ALIGN="LEFT" CELLPADDING="3">'
        f'<FONT COLOR="#2D3748" POINT-SIZE="10" FACE="Menlo">{it}</FONT>'
        f'</TD></TR>'
        for it in items
    )
    return header + rows


def _class_node(name: str, stereotype: str, icon: str,
                attrs: list[str], methods: list[str], variant: str) -> str:
    colors = PALETTE[variant]
    header = (
        f'<TR><TD BGCOLOR="{colors["header"]}" CELLPADDING="10">'
        f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
        f'<TR>'
        f'<TD FIXEDSIZE="TRUE" WIDTH="32" HEIGHT="32">'
        f'<IMG SRC="{icon}" SCALE="TRUE"/></TD>'
        f'<TD ALIGN="LEFT">'
        f'<FONT COLOR="white" POINT-SIZE="9"><I>«{stereotype}»</I></FONT><BR/>'
        f'<FONT COLOR="white" POINT-SIZE="15" FACE="Helvetica-Bold">  {name}</FONT>'
        f'</TD>'
        f'</TR>'
        f'</TABLE>'
        f'</TD></TR>'
    )
    body = (
        _section("atributos", attrs, colors["body"]) +
        _section("métodos",   methods, colors["body"])
    )
    return (
        f'<<TABLE BORDER="2" COLOR="{colors["border"]}" CELLBORDER="0" '
        f'CELLSPACING="0" CELLPADDING="0">'
        f'{header}{body}</TABLE>>'
    )


g = Digraph(
    "MyFraudLock_Clases",
    format="png",
    graph_attr={
        "label": "Diagrama de Clases  ·  MyFraudLock",
        "labelloc": "t",
        "fontname": "Helvetica-Bold",
        "fontsize": "22",
        "fontcolor": "#1A202C",
        "bgcolor": "#F7FAFC",
        "rankdir": "TB",
        "splines": "spline",
        "nodesep": "0.7",
        "ranksep": "1.0",
        "pad": "0.6",
        "compound": "true",
        "newrank": "true",
    },
    node_attr={"shape": "plaintext", "fontname": "Helvetica"},
    edge_attr={"fontname": "Helvetica", "fontsize": "10",
               "color": "#4A5568", "penwidth": "1.6"},
)

# ─── Capa Frontend ───────────────────────────────────────────────
with g.subgraph(name="cluster_frontend") as c:
    c.attr(label="Capa de Presentación  ·  Frontend",
           style="rounded,filled", bgcolor="#E6FFFA", color="#319795",
           fontcolor="#234E52", fontname="Helvetica-Bold", fontsize="14",
           penwidth="2", margin="16")
    c.node("Frontend", _class_node(
        "Frontend", "ui", ICON["js"],
        attrs=[
            "+ charts: ChartJS",
            "+ pollerInterval: int",
            "+ csrfToken: string",
        ],
        methods=[
            "+ graficos_financieros()",
            "+ grafico_shapley()",
            "+ long_polling_nuevos_incidentes()",
            "+ enviar_simulacion(payload)",
        ],
        variant="frontend",
    ))

# ─── Capa Web Controller ─────────────────────────────────────────
with g.subgraph(name="cluster_web") as c:
    c.attr(label="Capa Web  ·  Controladores Django",
           style="rounded,filled", bgcolor="#EBF8FF", color="#2B6CB0",
           fontcolor="#1A365D", fontname="Helvetica-Bold", fontsize="14",
           penwidth="2", margin="16")
    c.node("DjangoServer", _class_node(
        "DjangoServer", "framework", ICON["django"],
        attrs=["+ urls: urlpatterns", "+ middleware: list"],
        methods=["+ runserver()", "+ route(request)"],
        variant="controller",
    ))
    c.node("WebViews", _class_node(
        "WebViews", "view", ICON["django"],
        attrs=["+ context: dict"],
        methods=[
            "+ dashboard_view(request)",
            "+ incidentes_listado(request)",
            "+ incidente_detalle_view(request)",
            "+ simulacion_lote_view(request)",
        ],
        variant="view",
    ))
    c.node("TransaccionesAPI", _class_node(
        "TransaccionesAPI", "REST controller", ICON["django"],
        attrs=[
            "+ serializer: TransaccionSerializer",
            "+ permission_classes: list",
        ],
        methods=[
            "+ recibir_transaccion(JSON)",
            "+ validar_datos()",
            "+ enviar_a_ml()",
        ],
        variant="controller",
    ))

# ─── Capa de Servicios / ML ──────────────────────────────────────
with g.subgraph(name="cluster_services") as c:
    c.attr(label="Capa de Servicios  ·  Lógica de Negocio + ML",
           style="rounded,filled", bgcolor="#FAF5FF", color="#805AD5",
           fontcolor="#553C9A", fontname="Helvetica-Bold", fontsize="14",
           penwidth="2", margin="16")
    c.node("ModelLayerXAI", _class_node(
        "ModelLayerXAI", "ML service", ICON["ml"],
        attrs=[
            "- tf_model: tf.keras.Model",
            "- shap_explainer: DeepExplainer",
            "- threshold: float",
        ],
        methods=[
            "+ predict_fraud(tx: dict) → tuple",
            "+ predict_and_explain(payload: dict) → json",
            "- _ensure_artifacts()",
            "- _build_features(tx)",
        ],
        variant="ml",
    ))
    c.node("NotificationService", _class_node(
        "NotificationService", "service", ICON["python"],
        attrs=["+ smtp_host: str", "+ smtp_port: int"],
        methods=[
            "+ notify_new_incident(incidente)",
            "+ notify_fraud_confirmed(incidente)",
        ],
        variant="service",
    ))

# ─── Capa de Datos ───────────────────────────────────────────────
with g.subgraph(name="cluster_data") as c:
    c.attr(label="Capa de Datos  ·  Modelos ORM",
           style="rounded,filled", bgcolor="#F0FFF4", color="#38A169",
           fontcolor="#22543D", fontname="Helvetica-Bold", fontsize="14",
           penwidth="2", margin="16")
    c.node("DBHandler", _class_node(
        "DBHandler", "repository", ICON["pg"],
        attrs=["+ orm: django.db.models"],
        methods=[
            "+ guardar_transaccion(tx)",
            "+ generar_incidente(tx, score, shap)",
            "+ obtener_configuracion()",
        ],
        variant="data",
    ))

# ─── Relaciones UML ──────────────────────────────────────────────
# Rutas (Django Server -> Controllers)
g.edge("DjangoServer", "TransaccionesAPI",
       label="  routes (urls.py)  ", style="solid",
       arrowhead="vee", color=PALETTE["controller"]["header"])
g.edge("DjangoServer", "WebViews",
       label="  vistas autenticadas  ", style="solid",
       arrowhead="vee", color=PALETTE["view"]["header"])

# Renderizado
g.edge("WebViews", "Frontend",
       label="  renderiza contexto  ", style="dashed",
       arrowhead="vee", color=PALETTE["frontend"]["header"])

# Frontend -> API
g.edge("Frontend", "TransaccionesAPI",
       label="  POST /api/transacciones  ", style="dashed",
       arrowhead="vee", color="#2D3748")

# API -> ML
g.edge("TransaccionesAPI", "ModelLayerXAI",
       label="  llama inferencia  ", style="solid",
       arrowhead="vee", color=PALETTE["ml"]["header"], penwidth="2")

# ML -> DB
g.edge("ModelLayerXAI", "DBHandler",
       label="  score + SHAP  ", style="solid",
       arrowhead="vee", color=PALETTE["data"]["header"], penwidth="2")

# DB -> Notification
g.edge("DBHandler", "NotificationService",
       label="  trigger\nfraude  ", style="dashed",
       arrowhead="vee", color=PALETTE["service"]["header"])

# Notification -> Frontend (polling indirecto)
g.edge("WebViews", "DBHandler",
       label="  consulta ORM  ", style="solid",
       arrowhead="vee", color=PALETTE["data"]["header"])

g.render(OUT, cleanup=True)
print(f"Diagrama generado: {OUT}.png")
