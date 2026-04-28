"""
PDF → ZPL Converter Service v3.0
- Thread-safe: cada worker abre o PDF de forma independente
- Garante ^XZ em todas as etiquetas
- Suporte a lotes grandes sem truncamento
"""

import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF to ZPL Converter", version="3.0.0")

MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))


# ──────────────────────────────────────────
# Função de conversão (roda em worker separado)
# ──────────────────────────────────────────

def _converter_pagina(args: tuple) -> tuple:
    """
    Recebe (page_idx, pdf_bytes, width_inches, height_inches).
    Abre o PDF de forma independente — process-safe.
    Retorna (page_idx, zpl_string).
    """
    page_idx, pdf_bytes, width_inches, height_inches = args

    dpi = 203
    target_w = int(width_inches * dpi)
    target_h = int(height_inches * dpi)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_idx]

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)

    img = Image.frombytes("L", (pix.width, pix.height), bytes(pix.samples))

    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)

    img = img.convert("1", dither=Image.Dither.FLOYDSTEINBERG)

    width, height = img.size
    bytes_per_row = (width + 7) // 8

    hex_rows = []
    for y in range(height):
        row_bytes = []
        for x in range(0, width, 8):
            byte_val = 0
            for bit in range(8):
                if x + bit < width:
                    if img.getpixel((x + bit, y)) == 0:
                        byte_val |= (1 << (7 - bit))
            row_bytes.append(f"{byte_val:02X}")
        hex_rows.append("".join(row_bytes))

    zpl_hex = "".join(hex_rows)
    total_bytes = len(zpl_hex) // 2

    zpl = (
        f"^XA\n"
        f"^FO0,0\n"
        f"^GFA,{total_bytes},{total_bytes},{bytes_per_row},{zpl_hex}\n"
        f"^FS\n"
        f"^XZ\n"
    )

    doc.close()
    return (page_idx, zpl)


def pdf_para_zpl(pdf_bytes: bytes, width_inches: float, height_inches: float) -> str:
    """
    Converte todas as páginas em paralelo e retorna 1 string ZPL completa.
    """
    doc_check = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc_check)
    doc_check.close()

    logger.info(f"Iniciando: {n_pages} páginas | {width_inches}x{height_inches} pol | workers={MAX_WORKERS}")

    args_list = [
        (i, pdf_bytes, width_inches, height_inches)
        for i in range(n_pages)
    ]

    resultados = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_converter_pagina, args): args[0] for args in args_list}

        for future in as_completed(futures):
            try:
                idx, zpl = future.result()
                resultados[idx] = zpl
                done = len(resultados)
                if done % 10 == 0 or done == n_pages:
                    logger.info(f"  Progresso: {done}/{n_pages}")
            except Exception as e:
                idx = futures[future]
                logger.error(f"Erro na página {idx}: {e}")
                raise RuntimeError(f"Falha na página {idx}: {e}")

    partes = [resultados[i] for i in range(n_pages)]
    zpl_final = "".join(partes)

    n_xza = zpl_final.count("^XA")
    n_xzz = zpl_final.count("^XZ")
    logger.info(f"Concluído: {n_pages} páginas | ^XA={n_xza} | ^XZ={n_xzz} | {len(zpl_final)/1024/1024:.1f} MB")

    if n_xza != n_pages or n_xzz != n_pages:
        raise RuntimeError(f"ZPL inválido: esperado {n_pages} pares ^XA/^XZ, encontrado {n_xza}/{n_xzz}")

    return zpl_final


# ──────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0", "workers": MAX_WORKERS}


@app.post("/convert", response_class=PlainTextResponse)
async def convert(
    file: UploadFile = File(...),
    width_inches: float = Form(7.0),
    height_inches: float = Form(5.0),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Apenas arquivos .pdf são aceitos")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Arquivo vazio")

    logger.info(f"Recebido: {file.filename} | {len(pdf_bytes)/1024/1024:.1f} MB | {width_inches}x{height_inches} pol")

    try:
        zpl_content = pdf_para_zpl(pdf_bytes, width_inches, height_inches)
    except Exception as e:
        logger.error(f"Falha: {e}")
        raise HTTPException(500, str(e))

    n_labels = zpl_content.count("^XA")

    return PlainTextResponse(
        content=zpl_content,
        headers={
            "X-Labels-Count": str(n_labels),
            "X-Size-MB": f"{len(zpl_content)/1024/1024:.2f}",
            "X-Dimensions": f"{width_inches}x{height_inches}in",
        },
    )


@app.post("/convert/info")
async def convert_info(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    rect = page.rect
    info = {
        "filename": file.filename,
        "pages": len(doc),
        "size_mb": round(len(pdf_bytes) / 1024 / 1024, 2),
        "page_width_pt": round(rect.width, 1),
        "page_height_pt": round(rect.height, 1),
        "page_width_inches": round(rect.width / 72, 2),
        "page_height_inches": round(rect.height / 72, 2),
    }
    doc.close()
    return JSONResponse(info)
