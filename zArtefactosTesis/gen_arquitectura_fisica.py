"""
Genera el diagrama de Arquitectura Física de MyFraudLock con diseño moderno.
Incluye iconos reales: PostgreSQL, Django, Python, SHAP y el logo oficial
de Clever Cloud (PaaS donde se despliega la aplicación).
"""
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.framework import Django
from diagrams.programming.language import Python
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.client import Users, User
from diagrams.onprem.network import Internet
from diagrams.onprem.compute import Server
from diagrams.aws.ml import Sagemaker
from diagrams.generic.network import Firewall
from diagrams.generic.device import Tablet
from diagrams.generic.os import Ubuntu
from diagrams.custom import Custom
import os

BASE = os.path.dirname(__file__)
ICONS = os.path.join(BASE, "icons")
CLEVER_ICON = os.path.join(ICONS, "clever_cloud_square.png")
OUT = os.path.join(BASE, "arquitectura_fisica")

graph_attr = {
    "fontsize": "24",
    "fontname": "Helvetica-Bold",
    "bgcolor": "#F7FAFC",
    "pad": "0.8",
    "splines": "spline",
    "nodesep": "0.8",
    "ranksep": "1.4",
    "labelloc": "t",
    "compound": "true",
    "newrank": "true",
}

node_attr = {
    "fontsize": "13",
    "fontname": "Helvetica",
}

edge_attr = {
    "fontsize": "11",
    "fontname": "Helvetica",
    "color": "#4A5568",
}

cluster_internet = {
    "bgcolor": "#EDF2F7",
    "pencolor": "#718096",
    "fontcolor": "#2D3748",
    "fontsize": "15",
    "fontname": "Helvetica-Bold",
    "style": "rounded,filled",
    "penwidth": "2.5",
    "margin": "20",
}
# Cluster PaaS con colores corporativos de Clever Cloud (rojo coral)
cluster_paas = {
    "bgcolor": "#FFF5F5",
    "pencolor": "#C53030",
    "fontcolor": "#822727",
    "fontsize": "16",
    "fontname": "Helvetica-Bold",
    "style": "rounded,filled",
    "penwidth": "2.8",
    "margin": "25",
}
cluster_app = {
    "bgcolor": "#FAF5FF",
    "pencolor": "#805AD5",
    "fontcolor": "#553C9A",
    "fontsize": "14",
    "fontname": "Helvetica-Bold",
    "style": "rounded,filled",
    "penwidth": "2",
    "margin": "15",
}
cluster_data = {
    "bgcolor": "#F0FFF4",
    "pencolor": "#38A169",
    "fontcolor": "#22543D",
    "fontsize": "14",
    "fontname": "Helvetica-Bold",
    "style": "rounded,filled",
    "penwidth": "2",
    "margin": "15",
}
cluster_terceros = {
    "bgcolor": "#FFFAF0",
    "pencolor": "#DD6B20",
    "fontcolor": "#7B341E",
    "fontsize": "15",
    "fontname": "Helvetica-Bold",
    "style": "rounded,filled",
    "penwidth": "2.5",
    "margin": "20",
}

with Diagram(
    "Arquitectura Física  -  MyFraudLock  (Despliegue en Clever Cloud PaaS)",
    filename=OUT,
    show=False,
    direction="LR",
    outformat="png",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
):
    # ─── Red Pública ───────────────────────────────────────────────
    with Cluster("Red Pública  /  Internet", graph_attr=cluster_internet):
        usuarios = Users("Usuarios\n& Analistas")
        navegador = Tablet("Navegador Web\n/ Móvil")
        sistemas_ext = Server("Sistemas\nExternos (API)")
        usuarios >> Edge(style="dotted", color="#718096") >> navegador

    # ─── Servicios PaaS - Clever Cloud ────────────────────────────
    with Cluster(
        "Entorno PaaS  ·  Clever Cloud",
        graph_attr=cluster_paas,
    ):
        # Marca / branding del proveedor PaaS
        clever_brand = Custom("Clever Cloud\n(Proveedor PaaS)", CLEVER_ICON)

        firewall = Firewall("WAF / TLS\nLoad Balancer")

        with Cluster("Capa de Aplicación  (Runtime Python)", graph_attr=cluster_app):
            django_app = Django("Servidor Backend\nDjango + Gunicorn")
            with Cluster("Motor de Inferencia", graph_attr={
                "bgcolor": "#FFFFFF", "pencolor": "#D6BCFA",
                "style": "rounded,dashed", "fontsize": "12",
                "fontname": "Helvetica-Bold", "fontcolor": "#553C9A",
                "margin": "12",
            }):
                ml_engine = Sagemaker("Modelo\nDeep Learning")
                shap_engine = Python("SHAP\nExplainer")

        with Cluster("Capa de Datos Gestionada  ·  Clever Cloud Add-on", graph_attr=cluster_data):
            db_cluster = PostgreSQL("Clúster\nPostgreSQL")
            backup = Ubuntu("Backups\nAutomatizados")
            db_cluster >> Edge(style="dotted", color="#38A169", label=" snapshot diario ") >> backup

        # Indicar visualmente que Clever Cloud gestiona todos los servicios internos
        clever_brand >> Edge(
            style="dashed", color="#C53030", penwidth="1.5",
            label=" orquesta y\nescala "
        ) >> firewall
        clever_brand >> Edge(style="dashed", color="#C53030", penwidth="1.2") >> django_app
        clever_brand >> Edge(style="dashed", color="#C53030", penwidth="1.2") >> db_cluster

        firewall >> Edge(label=" Reenvío local ", color="#2B6CB0", penwidth="1.8") >> django_app
        django_app >> Edge(label=" Llamadas\nen memoria ", color="#805AD5", penwidth="1.8") >> ml_engine
        django_app >> Edge(label=" TCP / 5432\nSSL ", color="#38A169", penwidth="1.8") >> db_cluster

    # ─── Servicios de Terceros ─────────────────────────────────────
    with Cluster("Servicios de Terceros  /  TI Corporativa", graph_attr=cluster_terceros):
        smtp = Internet("Servidor SMTP\nCorporativo (TLS 587)")
        buzon = User("Buzón de\nAnalistas")
        smtp >> Edge(label=" entrega\ncorreo ", style="dashed", color="#DD6B20") >> buzon

    # ─── Conexiones de Borde ──────────────────────────────────────
    navegador >> Edge(label=" HTTPS  ·  443 ", color="#2D3748", penwidth="2.2", style="bold") >> firewall
    sistemas_ext >> Edge(label=" HTTPS  ·  443\n(API REST) ", color="#2D3748", penwidth="2.2", style="bold") >> firewall
    django_app >> Edge(label=" SMTP / TLS\nPuerto 587 ", color="#DD6B20", penwidth="1.8", style="bold") >> smtp

print(f"Diagrama generado: {OUT}.png")
