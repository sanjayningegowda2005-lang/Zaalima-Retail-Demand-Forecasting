import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = docx.Document()

title = doc.add_heading('Project 3 Roadmap: Data Pipeline Branch', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph()
p.add_run('Branch: ').bold = True
p.add_run('feature/data-pipeline\n')
p.add_run('Focus: ').bold = True
p.add_run('Cloud Data Warehouse, Data Ingestion & dbt Data Marts\n')
p.add_run('Timeline: ').bold = True
p.add_run('Days 1 – 20')

headers = ['Day', 'Task & Deliverables', 'Git Action']
data = [
    ["Day 1", "Provision Google BigQuery/Snowflake DW environment.", "git commit -m \"feat(dw): initialize bigquery dataset schemas\""],
    ["Day 2", "Download M5 dataset and write raw CSV ingestion scripts.", "git commit -m \"feat(ingest): add raw m5 csv loader scripts\""],
    ["Day 3", "Execute raw data quality checks (null handling, date formatting).", "git commit -m \"test(ingest): add raw data validation checks\""],
    ["Day 4", "Initialize dbt project and configure database connection.", "git commit -m \"feat(dbt): initialize dbt project configuration\""],
    ["Day 5", "Create primary dbt staging models (stg_sales, stg_calendar).", "git commit -m \"feat(dbt): build staging models for sales and calendar\""],
    ["Day 6", "Build dbt intermediate models to handle store-level aggregations.", "git commit -m \"feat(dbt): build store and department intermediate views\""],
    ["Day 7", "Build dbt analytics marts (mart_weekly_sales, mart_monthly_sales).", "git commit -m \"feat(dbt): create weekly and monthly analytics data marts\""],
    ["Day 8", "Add dbt schema validations (not_null, unique, relationships).", "git commit -m \"test(dbt): configure schema tests on analytics marts\""],
    ["Day 9", "Write dbt documentation and generate data lineage graphs.", "git commit -m \"docs(dbt): add model descriptions and lineage docs\""],
    ["Day 10", "PR & Merge Day 1: Merge feature/data-pipeline into main.", "git merge feature/data-pipeline"],
    ["Day 11", "Pull updated main into feature/data-pipeline. Create DB views for predictions.", "git commit -m \"feat(db): set up prediction logging tables and views\""],
    ["Day 12", "Create automated SQL script to log forecast outputs vs. actuals.", "git commit -m \"feat(sql): add forecast vs actual automated logging script\""],
    ["Day 13", "Optimize DW indexing / partitioning on forecast result tables.", "git commit -m \"perf(sql): add table partitioning on sales date columns\""],
    ["Day 14", "Write SQL scripts to calculate inventory safety stock levels.", "git commit -m \"feat(inventory): implement safety stock calculation logic\""],
    ["Day 15", "Add automated schema checks for incoming predictions.", "git commit -m \"test(db): enforce constraint checks on model output tables\""],
    ["Day 16", "Refactor dbt code and verify data pipeline execution logs.", "git commit -m \"refactor(dbt): optimize mart queries and pipeline execution\""],
    ["Day 17", "Sync branch with main after Phase 2 merge.", "git checkout main && git pull"],
    ["Day 18", "Write Dockerfile and .dockerignore for application containerization.", "git commit -m \"feat(docker): add container configuration for pipeline\""],
    ["Day 19", "Finalize root README.md, tech stack specs, and system architecture docs.", "git commit -m \"docs(readme): update system architecture and stack specs\""],
    ["Day 20", "Tag release v1.0.0 and complete project submission.", "git tag -a v1.0.0 -m \"Release v1.0.0\""]
]

table = doc.add_table(rows=1, cols=3)
table.style = 'Table Grid'

hdr_cells = table.rows[0].cells
for i, header_text in enumerate(headers):
    hdr_cells[i].text = header_text
    hdr_cells[i].paragraphs[0].runs[0].font.bold = True

for row in data:
    row_cells = table.add_row().cells
    for i, val in enumerate(row):
        row_cells[i].text = val

doc.save('Project3_Data_Pipeline_Branch.docx')
print("Successfully generated 'Project3_Data_Pipeline_Branch.docx'")