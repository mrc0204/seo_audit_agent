"""Generates input files separated strictly by Question (Q1, Q2, Q3)."""

from __future__ import annotations

import sys
import zipfile
import os
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from generate_test_deliverables import TEST_CASES, TestCase, generate_excel_inputs, generate_docx_inputs, generate_txt_inputs, create_outputs_zip

sys.stdout.reconfigure(encoding="utf-8")

if __name__ == "__main__":
    print("Generating input files separated by Q1, Q2, and Q3...")

    # Excel
    excel_path = "outputs/test_inputs.xlsx"
    try:
        generate_excel_inputs(TEST_CASES, excel_path)
    except PermissionError:
        excel_path = "outputs/test_inputs_by_question.xlsx"
        generate_excel_inputs(TEST_CASES, excel_path)

    # Word
    docx_path = "outputs/test_inputs.docx"
    try:
        generate_docx_inputs(TEST_CASES, docx_path)
    except PermissionError:
        docx_path = "outputs/test_inputs_by_question.docx"
        generate_docx_inputs(TEST_CASES, docx_path)

    # Text
    txt_path = "outputs/test_inputs.txt"
    generate_txt_inputs(TEST_CASES, txt_path)

    # ZIP archive update
    create_outputs_zip("outputs", "outputs/all_test_outputs.zip")
    print("All separated input files generated successfully!")
