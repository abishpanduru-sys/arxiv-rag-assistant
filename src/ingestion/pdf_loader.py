import pymupdf


def extract_pages(pdf_path):
    pdf = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(pdf):
        pages.append({
            "page_number": page_number + 1,
            "text": page.get_text()
        })

    pdf.close()

    return pages