"""
Genera el diagrama de Arquitectura Lógica de MyFraudLock con diseño moderno
usando iconos oficiales (PostgreSQL, Django, Python, SHAP, etc.).
"""
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.framework import Django
from diagrams.programming.language import Python, JavaScript
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.client import Users, Client
from diagrams.onprem.network import Internet
from diagrams.onprem.security import Vault
from diagrams.generic.storage import Storage
from diagrams.aws.ml import Sagemaker
import os

OUT = os.path.join(os.path.dirname(__file__), "arquitectura_logica")

graph_attr = {
    "fontsize": "24",
    "fontname": "Helvetica-Bold",
    "bgcolor": "#F7FAFC",
    "pad": "0.8",
    "splines": "ortho",
    "nodesep": "0.6",
    "ranksep": "1.2",
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

cluster_presentacion = {
    "bgcolor": "#EBF8FF",
    "pencolor": "#3182CE",
    "fontcolor": "#2C5282",
    "fontsize": "16",
    "fontname": "Helvetica-Bold",
    "style": "rounded,filled",
    "penwidth": "2.5",
    "margin": "20",
}
cluster_negocio = {
    "bgcolor": "#FAF5FF",
    "pencolor": "#805AD5",
    "fontcolor": "#553C9A",
    "fontsize": "16",
    "fontname": "Helvetica-Bold",
    "style": "rounded,filled",
    "penwidth": "2.5",
    "margin": "20",
}
cluster_ml = {
    "bgcolor": "#FFF5F5",
    "pencolor": "#E53E3E",
    "fontcolor": "#9B2C2C",
    "fontsize": "14",
    "fontname": "Helvetica-Bold",
    "style": "rounded,filled",
    "penwidth": "2",
    "margin": "15",
}
cluster_datos = {
    "bgcolor": "#F0FFF4",
    "pencolor": "#38A169",
    "fontcolor": "#22543D",
    "fontsize": "16",
    "fontname": "Helvetica-Bold",
    "style": "rounded,filled",
    "penwidth": "2.5",
    "margin": "20",
}
sub_cluster = {
    "bgcolor": "#FFFFFF",
    "pencolor": "#CBD5E0",
    "style": "rounded,dashed",
    "fontsize": "12",
    "fontname": "Helvetica-Bold",
    "fontcolor": "#4A5568",
    "margin": "12",
}

with Diagram(
    "Arquitectura Lógica  -  MyFraudLock",
    filename=OUT,
    show=False,
    direction="TB",
    outformat="png",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
):
    usuarios = Users("Usuarios / Analistas\n/ API Externa")

    with Cluster("CAPA DE PRESENTACIÓN  (Frontend)", graph_attr=cluster_presentacion):
        with Cluster("Módulos Funcionales", graph_attr=sub_cluster):
            dashboard = Client("Dashboard")
            incidentes_ui = Client("Incidentes")
            simulacion = Client("Simulación\nde Riesgo")
            config_ui = Client("Configuración")
            reportes = Client("Reportes PDF")
        ui = JavaScript("Interfaz de Usuario\nHTML5 · CSS3 · VanillaJS")
        ui >> Edge(style="dashed", color="#3182CE", penwidth="1.5") >> [
            dashboard, incidentes_ui, simulacion, config_ui, reportes
        ]

    with Cluster("CAPA DE NEGOCIO  (Backend - API REST)", graph_attr=cluster_negocio):
        router = Django("Enrutador REST\n(Django Views)")

        with Cluster("Servicios de Aplicación", graph_attr=sub_cluster):
            auth = Vault("Gestión de\nIdentidad y Sesión")
            ctrl_incidentes = Python("Controlador\nde Incidentes")
            notif = Internet("Motor de\nNotificaciones SMTP")

        with Cluster("Módulo Inteligente  (Machine Learning)", graph_attr=cluster_ml):
            modelo_dl = Sagemaker("Modelo\nDeep Learning")
            shap = Python("Explicabilidad\n(SHAP)")
            modelo_dl >> Edge(style="dotted", color="#E53E3E", penwidth="1.8") >> shap

        router >> Edge(label=" autenticar ", color="#805AD5", penwidth="1.5") >> auth
        router >> Edge(label=" gestionar ", color="#805AD5", penwidth="1.5") >> ctrl_incidentes
        router >> Edge(label=" notificar ", color="#805AD5", penwidth="1.5") >> notif
        ctrl_incidentes >> Edge(label=" inferir / explicar ", color="#E53E3E", penwidth="1.8", style="bold") >> modelo_dl

    with Cluster("CAPA DE PERSISTENCIA  (Datos)", graph_attr=cluster_datos):
        with Cluster("Modelo de Datos", graph_attr=sub_cluster):
            t_tx = Storage("Transacciones")
            t_usr = Storage("Usuarios")
            t_inc = Storage("Incidentes")
            t_cfg = Storage("Configuración")
        db = PostgreSQL("PostgreSQL")
        db >> Edge(style="dashed", color="#38A169", penwidth="1.5") >> [
            t_tx, t_usr, t_inc, t_cfg
        ]

    usuarios >> Edge(label="HTTP / HTTPS", color="#2D3748", penwidth="2", style="bold") >> ui
    [dashboard, incidentes_ui, simulacion, config_ui, reportes] >> Edge(
        label="JSON  ·  HTML", color="#2D3748", penwidth="2", style="bold"
    ) >> router
    ctrl_incidentes >> Edge(label="ORM Django  ·  SQL", color="#2D3748", penwidth="2", style="bold") >> db

print(f"Diagrama generado: {OUT}.png")
