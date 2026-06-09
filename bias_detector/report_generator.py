# bias_detector/report_generator.py
# Generates a clean formatted PDF audit report.
# reportlab is optional — install with: pip install reportlab

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
        PageBreak, Table, TableStyle
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from datetime import datetime


def generate_pdf(report, output_path="audit_report.pdf"):
    if not REPORTLAB_AVAILABLE:
        print(
            "\n  PDF generation skipped — reportlab is not installed.\n"
            "  To enable PDF reports run: pip install reportlab"
        )
        return None

    # ── Colors ───────────────────────────────────────────────────────────────
    NAVY   = colors.HexColor("#1E3A5F")
    DGRAY  = colors.HexColor("#24292F")
    MGRAY  = colors.HexColor("#57606A")
    LGRAY  = colors.HexColor("#F4F6F8")
    BORDER = colors.HexColor("#D0D7DE")
    FLAG   = colors.HexColor("#CF222E")
    OK     = colors.HexColor("#1A7F37")
    WHITE  = colors.white
    FLAGBG = colors.HexColor("#FFF5F5")

    # ── Styles ───────────────────────────────────────────────────────────────
    S = {}
    S["title"] = ParagraphStyle("title",
        fontSize=24, textColor=DGRAY, fontName="Helvetica-Bold",
        spaceAfter=6, spaceBefore=0, alignment=TA_CENTER)
    S["subtitle"] = ParagraphStyle("subtitle",
        fontSize=11, textColor=MGRAY, fontName="Helvetica",
        spaceAfter=6, spaceBefore=0, alignment=TA_CENTER)
    S["meta"] = ParagraphStyle("meta",
        fontSize=9, textColor=MGRAY, fontName="Helvetica",
        spaceAfter=0, alignment=TA_CENTER)
    S["sec"] = ParagraphStyle("sec",
        fontSize=10, textColor=WHITE, fontName="Helvetica-Bold",
        spaceAfter=0, spaceBefore=0, backColor=NAVY,
        leftIndent=6, rightIndent=6)
    S["ssh"] = ParagraphStyle("ssh",
        fontSize=10, textColor=DGRAY, fontName="Helvetica-Bold",
        spaceAfter=2, spaceBefore=8)
    S["body"] = ParagraphStyle("body",
        fontSize=9, textColor=DGRAY, fontName="Helvetica",
        spaceAfter=3, leading=13)
    S["flag"] = ParagraphStyle("flag",
        fontSize=9, textColor=FLAG, fontName="Helvetica-Bold",
        spaceAfter=2, leading=13)
    S["ok"] = ParagraphStyle("ok",
        fontSize=9, textColor=OK, fontName="Helvetica",
        spaceAfter=2, leading=13)
    S["disc"] = ParagraphStyle("disc",
        fontSize=7.5, textColor=MGRAY, fontName="Helvetica-Oblique",
        spaceAfter=2, leading=11, alignment=TA_CENTER)
    S["foot"] = ParagraphStyle("foot",
        fontSize=7.5, textColor=MGRAY, fontName="Helvetica",
        alignment=TA_CENTER)

    def rule():
        return HRFlowable(width="100%", thickness=0.4,
                          color=BORDER, spaceAfter=5, spaceBefore=5)

    def sec_hdr(title):
        return [Spacer(1, 8), Paragraph(title, S["sec"]), Spacer(1, 6)]

    def make_table(rows, col_widths, flag_rows=None):
        t = Table(rows, colWidths=col_widths, repeatRows=1)
        ts = [
            ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("TEXTCOLOR",     (0, 1), (-1, -1), DGRAY),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LGRAY, WHITE]),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("ALIGN",         (0, 1), (0, -1), "LEFT"),
            ("GRID",          (0, 0), (-1, -1), 0.25, BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ]
        if flag_rows:
            for r in flag_rows:
                ts += [
                    ("BACKGROUND", (0, r), (-1, r), FLAGBG),
                    ("TEXTCOLOR",  (0, r), (-1, r), FLAG),
                    ("FONTNAME",   (0, r), (-1, r), "Helvetica-Bold"),
                ]
        t.setStyle(TableStyle(ts))
        return t

    # ── Setup ────────────────────────────────────────────────────────────────
    config        = report.get("audit_config", {})
    attrs         = config.get("sensitive_attributes", [])
    threshold     = config.get("threshold", 0.1)
    results       = report.get("results_by_attribute", {})
    dataset_audit = report.get("dataset_audit", {})
    audit_date    = datetime.now().strftime("%B %d, %Y")

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.65*inch,
    )

    story = []

    # ── COVER PAGE ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("ML Bias Detector", S["title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Fairness Audit Report", S["subtitle"]))
    story.append(Spacer(1, 10))
    story.append(rule())
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Model: {report.get('model_name','Unknown')}     "
        f"Dataset: {report.get('dataset_name','Unknown')}     "
        f"Date: {audit_date}",
        S["meta"]
    ))
    story.append(Spacer(1, 4))
    story.append(rule())
    story.append(Spacer(1, 10))

    # Audit summary
    story.append(Paragraph("Audit Summary", S["ssh"]))
    story.append(Paragraph(
        f"This report presents the results of a fairness audit on the "
        f"<b>{report.get('model_name','')}</b> model using the "
        f"<b>{report.get('dataset_name','')}</b> dataset. "
        f"Sensitive attributes evaluated: <b>{', '.join(attrs)}</b>. "
        f"Any disparity exceeding <b>{threshold}</b> was flagged for review.",
        S["body"]
    ))
    story.append(Spacer(1, 6))

    # Overview table
    ov = [["Attribute", "Groups Evaluated", "Flagged Disparities"]]
    for attr, data in results.items():
        ov.append([attr,
                   str(len(data.get("group_metrics", {}))),
                   str(len(data.get("flagged_results", [])))])
    story.append(make_table(ov, [2.2*inch, 2*inch, 2.2*inch]))
    story.append(Spacer(1, 10))

    # Key findings
    story.append(Paragraph("Key Findings", S["ssh"]))
    for attr, data in results.items():
        flagged = data.get("flagged_results", [])
        groups  = list(data.get("group_metrics", {}).keys())
        if flagged:
            top = flagged[0]
            story.append(Paragraph(
                f"<b>{attr.upper()}:</b> {len(flagged)} disparity flag(s) across "
                f"{len(groups)} groups. Largest gap — {top['metric'].upper()} "
                f"between <b>{top['group_a']}</b> and <b>{top['group_b']}</b>: "
                f"{top['disparity']} (threshold {threshold}).",
                S["flag"]
            ))
        else:
            story.append(Paragraph(
                f"<b>{attr.upper()}:</b> No disparities exceeded the threshold "
                f"across {len(groups)} groups.", S["ok"]
            ))

    # Dataset health
    if dataset_audit:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Dataset Health Check", S["ssh"]))
        for attr, data in dataset_audit.items():
            flags = data.get("flags", [])
            if flags:
                types = ', '.join(sorted(set(f['check'] for f in flags)))
                story.append(Paragraph(
                    f"<b>{attr.upper()}:</b> {len(flags)} issue(s) detected — {types}.",
                    S["flag"]
                ))
            else:
                story.append(Paragraph(
                    f"<b>{attr.upper()}:</b> No issues detected.", S["ok"]
                ))

    story.append(Spacer(1, 12))
    story.append(rule())
    story.append(Paragraph(report.get("disclaimer", ""), S["disc"]))
    story.append(PageBreak())

    # ── DATASET AUDIT DETAIL ─────────────────────────────────────────────────
    story += sec_hdr("DATASET AUDIT — DETAILED RESULTS")
    for attr, data in dataset_audit.items():
        story.append(Paragraph(f"Attribute: {attr}", S["ssh"]))
        flags = data.get("flags", [])
        if flags:
            for f in flags:
                story.append(Paragraph(
                    f"• [{f['check'].upper()}] {f['message']}", S["flag"]))
        else:
            story.append(Paragraph("No issues detected.", S["ok"]))
        story.append(Spacer(1, 4))
    story.append(PageBreak())

    # ── MODEL AUDIT DETAIL ───────────────────────────────────────────────────
    for attr, data in results.items():
        story += sec_hdr(f"MODEL AUDIT — {attr.upper()}")

        gm      = data.get("group_metrics", {})
        flagged = data.get("flagged_results", [])

        # metrics table
        story.append(Paragraph("Per-Group Metrics", S["ssh"]))
        mh = ["Group","n","Accuracy","TPR","FPR","FNR","PPR","Precision","F1"]
        mr = [mh] + [
            [str(g), str(m.get("group_size","")), str(m.get("accuracy","")),
             str(m.get("tpr","")), str(m.get("fpr","")), str(m.get("fnr","")),
             str(m.get("ppr","")), str(m.get("precision","")), str(m.get("f1",""))]
            for g, m in gm.items()
        ]
        cw = [1.3*inch,.38*inch,.72*inch,.62*inch,.62*inch,.62*inch,.62*inch,.82*inch,.62*inch]
        story.append(make_table(mr, cw))
        story.append(Spacer(1, 8))

        # confusion matrix
        story.append(Paragraph("Confusion Matrix Values", S["ssh"]))
        ch = ["Group","TP","TN","FP","FN"]
        cr = [ch] + [
            [str(g), str(m.get("TP","")), str(m.get("TN","")),
             str(m.get("FP","")), str(m.get("FN",""))]
            for g, m in gm.items()
        ]
        story.append(make_table(cr, [2*inch,.8*inch,.8*inch,.8*inch,.8*inch]))
        story.append(Spacer(1, 8))

        # flagged disparities
        story.append(Paragraph("Flagged Disparities", S["ssh"]))
        if flagged:
            fh = ["Metric","Group A","Group B","Value A","Value B","Disparity","Threshold"]
            fr = [fh] + [
                [f["metric"].upper(), str(f["group_a"]), str(f["group_b"]),
                 str(f["value_a"]) if f["value_a"] is not None else "—",
                 str(f["value_b"]) if f["value_b"] is not None else "—",
                 str(f["disparity"]), str(f["threshold"])]
                for f in flagged
            ]
            story.append(make_table(fr,
                [.9*inch,1.1*inch,1.1*inch,.7*inch,.7*inch,.78*inch,.7*inch],
                flag_rows=list(range(1, len(fr)))))
        else:
            story.append(Paragraph(
                "No disparities exceeded the threshold.", S["ok"]))

        story.append(PageBreak())

    # ── AUDIT CONFIG ─────────────────────────────────────────────────────────
    story += sec_hdr("AUDIT CONFIGURATION")
    story.append(Paragraph(
        "The settings below were used for this audit. "
        "The same config will always produce the same report.",
        S["body"]
    ))
    story.append(Spacer(1, 6))

    cfg = [
        ["Setting", "Value"],
        ["Model Name",            report.get("model_name","")],
        ["Model Path",            report.get("model_path","")],
        ["Dataset Name",          report.get("dataset_name","")],
        ["Dataset Path",          report.get("dataset_path","")],
        ["Sensitive Attributes",  ", ".join(config.get("sensitive_attributes",[]))],
        ["Label Column",          config.get("label_column","")],
        ["Positive Label",        str(config.get("positive_label",""))],
        ["Threshold",             str(config.get("threshold",""))],
        ["Metrics Computed",      ", ".join(config.get("metrics_computed",[]))],
    ]
    story.append(make_table(cfg, [2.1*inch, 4.5*inch]))
    story.append(Spacer(1, 14))
    story.append(rule())
    story.append(Paragraph(report.get("disclaimer",""), S["disc"]))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        f"Generated by ml-bias-detector  |  {audit_date}", S["foot"]))

    # ── Page footer ──────────────────────────────────────────────────────────
    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MGRAY)
        canvas.drawString(0.75*inch, 0.38*inch,
            f"ml-bias-detector  |  {report.get('model_name','')}  |  {audit_date}")
        canvas.drawRightString(
            letter[0]-0.75*inch, 0.38*inch, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"  PDF report saved to: {output_path}")
    return output_path