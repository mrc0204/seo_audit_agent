"""Generate Exact Deliverables for Q1, Q2, and Q3 as requested.

Outputs:
- Q1_On_Page_Auditor/ -> only `audit.json` files
- Q2_NAP_Consistency_Checker/ -> only `nap_report.json` files
- Q3_Grounded_QA_Agent/ -> only `answer.json` files

Inputs:
- Excel (.xlsx), Word (.docx), and Text (.txt) formatted strictly matching Image 2 table headers:
  [Test ID, Category / Scope, Target Website URL, User Question (Q3), Description & Objective]
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from dataclasses import dataclass

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from app.main import build_llm_generate, run_pipeline

sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class QuestionTestCase:
    test_id: str
    question_num: str  # "Q1", "Q2", "Q3"
    category: str      # "Q1 (SEO Audit Findings)", "Q2 (NAP Consistency)", "Q3 (Grounded QA)"
    target_url: str
    question: str | None
    description: str
    max_pages: int = 15
    max_depth: int = 3


TEST_SUITE: list[QuestionTestCase] = [
    # --- Question 3: Grounded Q&A Agent ---
    QuestionTestCase(
        test_id="TC01",
        question_num="Q3",
        category="Q3 (Grounded QA)",
        target_url="https://webflow.com/",
        question="Can webflow is used for website development?",
        description="Verify webflow website development capabilities in QA agent.",
    ),
    QuestionTestCase(
        test_id="TC02",
        question_num="Q3",
        category="Q3 (Grounded QA)",
        target_url="https://www.python.org",
        question="What is python?",
        description="Verify general definition/overview answering for python.org.",
    ),
    QuestionTestCase(
        test_id="TC03",
        question_num="Q2",
        category="Q2 (NAP Consistency)",
        target_url="https://ccbp.in/",
        question=None,
        description="Check Name, Address, Phone (NAP) consistency across ccbp.in pages.",
    ),
    QuestionTestCase(
        test_id="TC04",
        question_num="Q2",
        category="Q2 (NAP Consistency)",
        target_url="https://www.niatindia.com/",
        question=None,
        description="Check Name, Address, Phone (NAP) consistency across niatindia.com pages.",
    ),
    QuestionTestCase(
        test_id="TC05",
        question_num="Q2",
        category="Q2 (NAP Consistency)",
        target_url="https://www.thehudsonkitchen.com/",
        question=None,
        description="Check Name, Address, Phone (NAP) consistency across local business pages of Hudson Kitchen.",
    ),
    QuestionTestCase(
        test_id="TC06",
        question_num="Q3",
        category="Q3 (Grounded QA)",
        target_url="https://ccbp.in/",
        question="What is this domain for?",
        description="Extract evidence and explain core purpose of ccbp.in domain.",
    ),
    QuestionTestCase(
        test_id="TC07",
        question_num="Q3",
        category="Q3 (Grounded QA)",
        target_url="https://www.niatindia.com/",
        question="What programs or courses are offered by NIAT?",
        description="Extract course/program offerings with grounded citations from NIAT India.",
    ),
    QuestionTestCase(
        test_id="TC08",
        question_num="Q3",
        category="Q3 (Grounded QA)",
        target_url="https://www.thehudsonkitchen.com/",
        question="Where is Hudson Kitchen located and what services do they provide?",
        description="Answer location and service details for Hudson Kitchen using extracted page text.",
    ),
    QuestionTestCase(
        test_id="TC09",
        question_num="Q1",
        category="Q1 (SEO Audit Findings)",
        target_url="https://webflow.com/",
        question=None,
        description="Run technical & on-page SEO audit findings against webflow.com homepage & subpages.",
    ),
    QuestionTestCase(
        test_id="TC10",
        question_num="Q1",
        category="Q1 (SEO Audit Findings)",
        target_url="https://www.python.org",
        question=None,
        description="Run technical & on-page SEO audit findings against python.org.",
    ),
    QuestionTestCase(
        test_id="TC11",
        question_num="Q1",
        category="Q1 (SEO Audit Findings)",
        target_url="https://ccbp.in/",
        question=None,
        description="Run technical & on-page SEO audit findings against ccbp.in.",
    ),
    QuestionTestCase(
        test_id="TC12",
        question_num="Q1",
        category="Q1 (SEO Audit Findings)",
        target_url="https://www.niatindia.com/",
        question=None,
        description="Run technical & on-page SEO audit findings against niatindia.com.",
    ),
    QuestionTestCase(
        test_id="TC13",
        question_num="Q1",
        category="Q1 (SEO Audit Findings)",
        target_url="https://www.thehudsonkitchen.com/",
        question=None,
        description="Run technical & on-page SEO audit findings against thehudsonkitchen.com.",
    ),
]


def run_question_specific_tests(base_output_dir: str = "outputs") -> None:
    q1_dir = os.path.join(base_output_dir, "Q1_On_Page_Auditor")
    q2_dir = os.path.join(base_output_dir, "Q2_NAP_Consistency_Checker")
    q3_dir = os.path.join(base_output_dir, "Q3_Grounded_QA_Agent")

    os.makedirs(q1_dir, exist_ok=True)
    os.makedirs(q2_dir, exist_ok=True)
    os.makedirs(q3_dir, exist_ok=True)

    llm_key = os.environ.get("LLM_API_KEY")
    llm_gen = build_llm_generate("nvidia", llm_key)

    print("=" * 75)
    print(f"Executing Question-Specific Runs ({len(TEST_SUITE)} test cases)...")
    print("=" * 75)

    for tc in TEST_SUITE:
        print(f"\nProcessing [{tc.test_id}] {tc.category} for {tc.target_url}")
        domain_slug = tc.target_url.replace("https://", "").replace("http://", "").replace("www.", "").strip("/").replace("/", "_")

        run_q1 = tc.question_num == "Q1"
        run_q2 = tc.question_num == "Q2"
        run_q3 = tc.question_num == "Q3"

        res = run_pipeline(
            url=tc.target_url,
            question=tc.question if run_q3 else None,
            max_pages=tc.max_pages,
            max_depth=tc.max_depth,
            timeout=12,
            llm_generate=llm_gen,
            run_q1=run_q1,
            run_q2=run_q2,
            run_q3=run_q3,
        )

        if run_q1:
            out_file = os.path.join(q1_dir, f"{tc.test_id}_{domain_slug}_audit.json")
            audit_json = [f.to_audit_entry() for f in res.findings]
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(audit_json, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved Q1 audit.json ({len(audit_json)} findings) -> {out_file}")

        elif run_q2:
            out_file = os.path.join(q2_dir, f"{tc.test_id}_{domain_slug}_nap_report.json")
            nap_json = [c.to_report_entry() for c in res.nap_comparisons]
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(nap_json, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved Q2 nap_report.json ({len(nap_json)} fields) -> {out_file}")

        elif run_q3:
            out_file = os.path.join(q3_dir, f"{tc.test_id}_{domain_slug}_answer.json")
            ans_json = res.answer.to_answer_json() if res.answer else {"query": tc.question, "url": None, "excerpt": None}
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(ans_json, f, indent=2, ensure_ascii=False)
            status_str = f"Found on {ans_json.get('url')}" if ans_json.get("url") else "Null (no exact match)"
            print(f"  ✓ Saved Q3 answer.json ({status_str}) -> {out_file}")


def generate_exact_excel_inputs(file_path: str) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Test Inputs"

    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    data_font = Font(name="Calibri", size=10)
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    headers = [
        "Test ID",
        "Category / Scope",
        "Target Website URL",
        "User Question (Q3)",
        "Description & Objective",
    ]
    ws.append(headers)

    for col_num in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_num)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for tc in TEST_SUITE:
        row = [
            tc.test_id,
            tc.category,
            tc.target_url,
            tc.question or "N/A (Crawl/Audit Only)",
            tc.description,
        ]
        ws.append(row)
        curr_row = ws.max_row
        for col_num in range(1, len(headers) + 1):
            c = ws.cell(row=curr_row, column=col_num)
            c.font = data_font
            c.border = thin_border
            if col_num == 1:
                c.alignment = Alignment(horizontal="center", vertical="center")

    widths = [12, 25, 36, 45, 60]
    for i, col_letter in enumerate(["A", "B", "C", "D", "E"]):
        ws.column_dimensions[col_letter].width = widths[i]

    wb.save(file_path)
    print(f"Generated exact Excel inputs file: {file_path}")


def generate_exact_docx_inputs(file_path: str) -> None:
    doc = docx.Document()

    p_title = doc.add_paragraph()
    run_title = p_title.add_run("Evidence-Grounded SEO Audit Agent — Test Inputs Matrix")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(16)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(31, 78, 120)

    table = doc.add_table(rows=1, cols=5)
    table.style = "Table Grid"

    hdr_cells = table.rows[0].cells
    hdr_titles = ["Test ID", "Category / Scope", "Target Website URL", "User Question (Q3)", "Description & Objective"]
    for i, title in enumerate(hdr_titles):
        hdr_cells[i].text = title
        shd = parse_xml(r'<w:shd {} w:fill="1F4E78"/>'.format(nsdecls('w')))
        hdr_cells[i]._tc.get_or_add_tcPr().append(shd)
        for p in hdr_cells[i].paragraphs:
            for run in p.runs:
                run.font.bold = True
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(255, 255, 255)

    for tc in TEST_SUITE:
        row_cells = table.add_row().cells
        row_cells[0].text = tc.test_id
        row_cells[1].text = tc.category
        row_cells[2].text = tc.target_url
        row_cells[3].text = tc.question or "N/A (Crawl/Audit Only)"
        row_cells[4].text = tc.description

        for cell in row_cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(8.5)

    doc.save(file_path)
    print(f"Generated exact Word inputs file: {file_path}")


def generate_exact_txt_inputs(file_path: str) -> None:
    lines = [
        "=========================================================================================",
        "                    EVIDENCE-GROUNDED SEO AUDIT AGENT — TEST INPUTS",
        "=========================================================================================\n",
    ]

    for tc in TEST_SUITE:
        lines.append(f"Test ID       : {tc.test_id}")
        lines.append(f"Category      : {tc.category}")
        lines.append(f"Target URL    : {tc.target_url}")
        lines.append(f"User Question : {tc.question or 'N/A (Crawl/Audit Only)'}")
        lines.append(f"Description   : {tc.description}")
        lines.append("-" * 89 + "\n")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated exact Text inputs file: {file_path}")


def zip_question_outputs(base_output_dir: str = "outputs", zip_filename: str = "outputs/all_test_outputs.zip") -> None:
    abs_zip_path = os.path.abspath(zip_filename)
    print(f"\nCreating ZIP archive: {abs_zip_path}")

    folders_to_include = [
        "Q1_On_Page_Auditor",
        "Q2_NAP_Consistency_Checker",
        "Q3_Grounded_QA_Agent",
    ]

    files_to_include = [
        "test_inputs.xlsx",
        "test_inputs.docx",
        "test_inputs.txt",
    ]

    with zipfile.ZipFile(abs_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include question folders
        for fld in folders_to_include:
            fld_path = os.path.join(base_output_dir, fld)
            if os.path.exists(fld_path):
                for root, _, files in os.walk(fld_path):
                    for file in files:
                        full_p = os.path.join(root, file)
                        rel_p = os.path.relpath(full_p, base_output_dir)
                        zf.write(full_p, rel_p)

        # Include input matrix files
        for fn in files_to_include:
            fp = os.path.join(base_output_dir, fn)
            if os.path.exists(fp):
                zf.write(fp, fn)

    print(f"Successfully created ZIP archive ({os.path.getsize(abs_zip_path)} bytes).")


if __name__ == "__main__":
    run_question_specific_tests("outputs")

    # Generate input files in exact table format
    excel_path = "outputs/test_inputs.xlsx"
    try:
        generate_exact_excel_inputs(excel_path)
    except PermissionError:
        excel_path = "outputs/test_inputs_matrix.xlsx"
        generate_exact_excel_inputs(excel_path)

    generate_exact_docx_inputs("outputs/test_inputs.docx")
    generate_exact_txt_inputs("outputs/test_inputs.txt")

    zip_question_outputs("outputs", "outputs/all_test_outputs.zip")
    print("\nAll deliverables generated successfully with exact Question-specific outputs!")
