📘 Automatic Question Paper Generator

📌 Project Description



The Automatic Question Paper Generator is an AI-based web application that automatically generates examination question papers from a syllabus PDF.

It extracts syllabus content, filters irrelevant text, classifies topics, generates meaningful questions using NLP techniques, stores them in a database, and produces a formatted question paper PDF.



🎯 Objectives



To reduce manual effort in question paper preparation



To automatically generate syllabus-based questions



To classify questions by type (definition, analysis, comparison)



To produce a printable question paper in PDF format



To avoid duplicate, irrelevant, and noisy questions



⚙️ Technologies Used



Python



Flask (Web Framework)



Transformers (T5 Model) – Question generation



PyPDF – PDF text extraction



SQLite – Question storage



ReportLab – PDF generation



Git \& GitHub – Version control



🏗️ System Architecture



User uploads syllabus PDF



PDF text is extracted and cleaned



Noise and headers are filtered



Topics are classified



Questions are generated using AI



Questions are stored in database



Question paper PDF is generated



✨ Features



Upload syllabus in PDF format



AI-based question generation



Duplicate question prevention



Noise and header filtering



Clean, exam-ready questions



Automatic PDF generation



GitHub-ready project structure

Markdown-first PDF parsing (noise reduction)

The app now supports a markdown-first PDF extraction path before concept extraction.

- Default mode (`PDF_PARSER=auto`): tries Docling markdown export, then Marker markdown, then falls back to PyPDF.
- Force a parser with `PDF_PARSER=docling`, `PDF_PARSER=marker`, or `PDF_PARSER=pypdf`.
- If Docling or Marker is not installed, the app automatically falls back to PyPDF.

