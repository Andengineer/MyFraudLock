"""
Utilidad para generar PDFs con ReportLab.
"""
import io
from datetime import datetime

from django.http import HttpResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Brand colours ────────────────────────────────────────────────────────────
BRAND     = colors.HexColor("#6c3eff")
BRAND_D   = colors.HexColor("#4318b8")
DANGER    = colors.HexColor("#ef4444")
WARNING   = colors.HexColor("#f59e0b")
SUCCESS   = colors.HexColor("#10b981")
GRAY_100  = colors.HexColor("#f3f4f6")
GRAY_700  = colors.HexColor("#374151")

# ── Shared styles ────────────────────────────────────────────────────────────
_styles = getSampleStyleSheet()

STYLE_TITLE = ParagraphStyle(
    "BrandTitle", parent=_styles["Title"],
    textColor=BRAND_D, fontSize=20, spaceAfter=4,
)
STYLE_H2 = ParagraphStyle(
    "BrandH2", parent=_styles["Heading2"],
    textColor=BRAND_D, fontSize=14, spaceAfter=6,
)
STYLE_BODY = ParagraphStyle(
    "BrandBody", parent=_styles["Normal"],
    fontSize=9, leading=13,
)
STYLE_SMALL = ParagraphStyle(
    "BrandSmall", parent=_styles["Normal"],
    fontSize=7.5, textColor=GRAY_700,
)

TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), BRAND),
    ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
    ("FONTSIZE",   (0, 0), (-1, 0), 8),
    ("FONTSIZE",   (0, 1), (-1, -1), 7.5),
    ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ("TOPPADDING",  (0, 1), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRAY_100]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
])


def _header(elements, title, subtitle=""):
    """Add branded header to elements list."""
    elements.append(Paragraph("MyFraudLock", STYLE_SMALL))
    elements.append(Paragraph(title, STYLE_TITLE))
    if subtitle:
        elements.append(Paragraph(subtitle, STYLE_BODY))
    elements.append(Spacer(1, 4 * mm))
    elements.append(
        HRFlowable(width="100%", thickness=1, color=BRAND, spaceAfter=6)
    )


def _footer_text():
    return f"Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')} — MyFraudLock"


def _build_pdf(filename, build_func):
    """Core builder: returns an HttpResponse with the PDF."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
    )

    elements = []
    build_func(elements)

    # Footer
    elements.append(Spacer(1, 10 * mm))
    elements.append(
        HRFlowable(width="100%", thickness=0.5, color=GRAY_700, spaceAfter=4)
    )
    elements.append(Paragraph(_footer_text(), STYLE_SMALL))

    doc.build(elements)
    buf.seek(0)

    response = HttpResponse(buf, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard_pdf(ctx):
    """Reporte PDF del dashboard."""
    def build(el):
        _header(el, "Reporte Dashboard", "Resumen ejecutivo de incidentes")

        # KPIs
        kpi_data = [
            ["Indicador", "Valor"],
            ["Pendientes", str(ctx["pendientes"])],
            ["Fraude confirmado", str(ctx["confirmados"])],
            ["Falsos positivos", str(ctx["falsos"])],
        ]
        t = Table(kpi_data, colWidths=[120 * mm, 40 * mm])
        t.setStyle(TABLE_STYLE)
        el.append(t)
        el.append(Spacer(1, 8 * mm))

        # Recent incidents
        el.append(Paragraph("Últimos incidentes", STYLE_H2))
        rows = [["ID", "Estado", "Score", "Fecha"]]
        for i in ctx.get("recientes", []):
            rows.append([
                f"#{i.id_incidente}",
                i.estado,
                f"{i.score_riesgo}%",
                i.fecha.strftime("%d/%m/%Y %H:%M"),
            ])
        if len(rows) > 1:
            t2 = Table(rows, colWidths=[25 * mm, 45 * mm, 30 * mm, 60 * mm])
            t2.setStyle(TABLE_STYLE)
            el.append(t2)
        else:
            el.append(Paragraph("No hay incidentes recientes.", STYLE_BODY))

    return _build_pdf("dashboard_reporte.pdf", build)


def build_incidentes_pdf(incidentes):
    """Reporte PDF de la lista de incidentes."""
    def build(el):
        _header(el, "Reporte de Incidentes", f"{incidentes.count()} registros")

        rows = [["ID", "Score", "Estado", "Fecha"]]
        for i in incidentes:
            rows.append([
                f"#{i.id_incidente}",
                f"{i.score_riesgo}%",
                i.estado,
                i.fecha.strftime("%d/%m/%Y %H:%M"),
            ])

        if len(rows) > 1:
            t = Table(rows, colWidths=[25 * mm, 30 * mm, 50 * mm, 55 * mm])
            t.setStyle(TABLE_STYLE)
            el.append(t)
        else:
            el.append(Paragraph("No hay incidentes.", STYLE_BODY))

    return _build_pdf("incidentes_reporte.pdf", build)


def build_incidente_detalle_pdf(incidente):
    """Reporte PDF de un incidente individual."""
    tx = incidente.id_transaccion

    def build(el):
        _header(
            el,
            f"Incidente #{incidente.id_incidente}",
            f"Estado: {incidente.estado} — Score: {incidente.score_riesgo}%",
        )

        # Details table
        info = [
            ["Campo", "Valor"],
            ["ID Incidente", str(incidente.id_incidente)],
            ["Estado", incidente.estado],
            ["Score riesgo", f"{incidente.score_riesgo}%"],
            ["Fecha", incidente.fecha.strftime("%d/%m/%Y %H:%M")],
            ["Comentario", incidente.comentario or "—"],
        ]
        t = Table(info, colWidths=[50 * mm, 110 * mm])
        t.setStyle(TABLE_STYLE)
        el.append(t)
        el.append(Spacer(1, 6 * mm))

        # Transaction
        el.append(Paragraph("Transacción asociada", STYLE_H2))
        tx_info = [
            ["Campo", "Valor"],
            ["ID Transacción", str(tx.id_transaccion)],
            ["Importe", f"S/ {tx.importe}"],
            ["Marca de Tarjeta", tx.card_brand.upper() if tx.card_brand else "—"],
            ["Tipo de Tarjeta", tx.card_type.title() if tx.card_type else "—"],
            ["Banco Emisor", tx.issuer_bank.upper() if tx.issuer_bank else "—"],
            ["Canal de Pago", tx.payment_channel.title() if tx.payment_channel else "—"],
            ["Código ECI", str(tx.eci_code)],
            ["Región", tx.customer_region.title() if tx.customer_region else "—"],
            ["Categoría", tx.product_category or tx.category or "—"],
            ["Nro. Ítems", str(tx.num_items)],
            ["Ratio Denegación (DAR)", str(tx.ratio_dar)],
            ["Ratio Monto/Mediana (CMR)", str(tx.ratio_cmr)],
        ]
        t2 = Table(tx_info, colWidths=[50 * mm, 110 * mm])
        t2.setStyle(TABLE_STYLE)
        el.append(t2)
        el.append(Spacer(1, 6 * mm))

        # Explainability
        if incidente.explicabilidad:
            el.append(Paragraph("Explicabilidad (SHAP)", STYLE_H2))
            try:
                import json
                exp = incidente.explicabilidad
                if isinstance(exp, str):
                    exp = json.loads(exp)
                factors = exp.get("top_factors", [])
                if factors:
                    rows = [["Factor", "Valor", "Impacto"]]
                    for f in factors:
                        factor_name = f.get("display_name", f.get("feature", "—"))
                        rows.append([
                            factor_name,
                            str(f.get("value", "—")),
                            f"{f.get('impact', 0):.4f}",
                        ])
                    t3 = Table(rows, colWidths=[65 * mm, 45 * mm, 50 * mm])
                    t3.setStyle(TABLE_STYLE)
                    el.append(t3)
                    
                    # Añadir las reglas de negocio si existen
                    negocio = exp.get("explicacion_negocio", [])
                    if negocio:
                        el.append(Spacer(1, 4 * mm))
                        el.append(Paragraph("Conclusión Funcional (Reglas)", STYLE_H2))
                        for regla in negocio:
                            el.append(Paragraph(f"• {regla}", STYLE_BODY))
                            el.append(Spacer(1, 2 * mm))
                else:
                    el.append(Paragraph(
                        str(exp)[:500], STYLE_SMALL
                    ))
            except Exception:
                el.append(Paragraph("Datos de explicabilidad no disponibles.", STYLE_BODY))

    return _build_pdf(f"incidente_{incidente.id_incidente}.pdf", build)
