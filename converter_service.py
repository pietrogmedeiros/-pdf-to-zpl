"""
PDF → ZPL Converter Service
Chamado pelo n8n via HTTP Request node
"""

import io
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF to ZPL Converter", version="2.1.0")

MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))

# ──────────────────────────────────────────
# Conversão core
# ──────────────────────────────────────────

def pagina_para_zpl(doc: fitz.Document, page_idx: int, width_inches: float, height_inches: float) -> str:
    """Converte uma página do PDF diretamente para ZPL, forçando o tamanho correto."""
    dpi = 203  # DPI padrão de impressoras Zebra
    target_width = int(width_inches * dpi)
    target_height = int(height_inches * dpi)

    page = doc[page_idx]

    # Renderiza a página como bitmap escala de cinza a 203 DPI
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)

    # Converte para imagem PIL
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)

    # Redimensiona para o tamanho exato da etiqueta configurada
    if img.size != (target_width, target_height):
        logger.info(f"  Redimensionando página {page_idx} de {img.size} para ({target_width}, {target_height})")
        img = img.resize((target_width, target_height), Image.LANCZOS)

    # Converte para 1-bit com dithering Floyd-Steinberg
    img = img.convert("1", dither=Image.Dither.FLOYDSTEINBERG)

    width, height = img.size
    bytes_per_row = (width + 7) // 8

    # Gera hex ZPL linha a linha
    hex_rows = []
    for y in range(height):
        row_bytes = []
        for x in range(0, width, 8):
            byte_val = 0
            for bit in range(8):
                if x + bit < width:
                    if img.getpixel((x + bit, y)) == 0:  # preto
                        byte_val |= (1 << (7 - bit))
            row_bytes.append(f"{byte_val:02X}")
        hex_rows.append("".join(row_bytes))

    zpl_hex = "".join(hex_rows)
    total_bytes = len(zpl_hex) // 2

    return (
        f"^XA\n"
        f"^FO0,0\n"
        f"^GFA,{total_bytes},{total_bytes},{bytes_per_row},{zpl_hex}\n"
        f"^FS\n"
        f"^XZ\n"
    )


def pdf_para_zpl(pdf_bytes: bytes, width_inches: float, height_inches: float) -> str:
    """
    Converte todas as páginas do PDF para um único arquivo ZPL.
    Usa processamento paralelo para velocidade máxima.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)

    logger.info(f"Convertendo {n_pages} páginas | {width_inches}x{height_inches} pol | {MAX_WORKERS} workers")

    resultados: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(pagina_para_zpl, doc, i, width_inches, height_inches): i
            for i in range(n_pages)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                resultados[idx] = future.result()
                if (idx + 1) % 10 == 0 or idx == n_pages - 1:
                    logger.info(f"  Progresso: {idx + 1}/{n_pages}")
            except Exception as e:
                logger.error(f"Erro na página {idx}: {e}")
                raise

    # Junta em ordem
    zpl_final = "".join(resultados[i] for i in range(n_pages))
    logger.info(f"Concluído: {n_pages} etiquetas → {len(zpl_final) / 1024 / 1024:.1f} MB")
    return zpl_final


# ──────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────

@app.get("/health")
def health():
    """n8n pode usar este endpoint para verificar se o serviço está rodando."""
    return {"status": "ok", "workers": MAX_WORKERS}


@app.post("/convert", response_class=PlainTextResponse)
async def convert(
    file: UploadFile = File(..., description="Arquivo PDF"),
    width_inches: float = Form(7.0, description="Largura em polegadas (ex: 7 ou 10)"),
    height_inches: float = Form(5.0, description="Altura em polegadas (ex: 5 ou 15)"),
):
    """
    Recebe um PDF e retorna o ZPL completo (todas as páginas, 1 arquivo).

    Uso no n8n:
      - Method: POST
      - URL: http://n8n_webco_pdf-converter:8000/convert
      - Body: Form Data
          file: [binary do PDF]
          width_inches: 7  (ou 10)
          height_inches: 5  (ou 15)
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas arquivos .pdf são aceitos")

    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Arquivo vazio")

    logger.info(f"Recebido: {file.filename} ({len(pdf_bytes) / 1024 / 1024:.1f} MB) | {width_inches}x{height_inches} pol")

    try:
        zpl_content = pdf_para_zpl(pdf_bytes, width_inches, height_inches)
    except Exception as e:
        logger.error(f"Falha na conversão: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return PlainTextResponse(
        content=zpl_content,
        headers={
            "X-Pages-Converted": str(zpl_content.count("^XA")),
            "X-File-Size-MB": f"{len(zpl_content) / 1024 / 1024:.2f}",
        },
    )


@app.post("/convert/info")
async def convert_info(file: UploadFile = File(...)):
    """Retorna info do PDF sem converter (útil para debug no n8n)."""
    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    rect = page.rect
    return JSONResponse({
        "filename": file.filename,
        "pages": len(doc),
        "size_mb": round(len(pdf_bytes) / 1024 / 1024, 2),
        "page_width_pt": round(rect.width, 1),
        "page_height_pt": round(rect.height, 1),
        "page_width_inches": round(rect.width / 72, 2),
        "page_height_inches": round(rect.height / 72, 2),
    })
