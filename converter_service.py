"""
PDF → ZPL Converter Service v4.0
- ThreadPoolExecutor com doc independente por thread (sem thread-safety issues)
- Endpoint /debug/render para inspecionar o PNG gerado
- Log de estatísticas de pixel para detectar imagem em branco
"""

import io
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse, Response
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF to ZPL Converter", version="4.0.0")

MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
DPI = int(os.getenv("DPI", 203))


# ──────────────────────────────────────────
# Core: renderiza 1 página → ZPL
# ──────────────────────────────────────────

def renderizar_pagina(pdf_bytes: bytes, page_idx: int, target_w: int, target_h: int) -> Image.Image:
    """Abre o PDF de forma independente e renderiza a página como imagem PIL grayscale."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_idx]

    # Renderiza em escala de cinza a DPI configurado
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)

    img = Image.frombytes("L", (pix.width, pix.height), bytes(pix.samples))
    doc.close()

    # Redimensiona para o tamanho exato da etiqueta
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)

    return img


def imagem_para_zpl(img: Image.Image) -> str:
    """Converte uma imagem PIL grayscale em bloco ZPL (^XA...^XZ)."""
    # Verifica se a imagem está muito clara (possível problema de renderização)
    pixels = list(img.getdata())
    min_px = min(pixels)
    max_px = max(pixels)
    logger.info(f"  Pixel stats: min={min_px} max={max_px} (escala cinza, 0=preto 255=branco)")

    if max_px - min_px < 10:
        logger.warning(f"  AVISO: imagem com pouco contraste (min={min_px} max={max_px}). Pode ser renderização em branco.")

    # Converte para 1-bit
    img_1bit = img.convert("1", dither=Image.Dither.FLOYDSTEINBERG)

    width, height = img_1bit.size
    bytes_per_row = (width + 7) // 8

    hex_rows = []
    for y in range(height):
        row_bytes = []
        for x in range(0, width, 8):
            byte_val = 0
            for bit in range(8):
                if x + bit < width:
                    # No modo "1" do PIL: 0 = preto, 255 = branco
                    if img_1bit.getpixel((x + bit, y)) == 0:
                        byte_val |= (1 << (7 - bit))
            row_bytes.append(f"{byte_val:02X}")
        hex_rows.append("".join(row_bytes))

    zpl_hex = "".join(hex_rows)
    total_bytes = len(zpl_hex) // 2

    # Verificação de sanidade: ZPL não pode ser all-zeros
    unique_chars = set(zpl_hex)
    if unique_chars == {"0"}:
        logger.warning("  AVISO: ZPL hex é todo zeros — imagem renderizou como branco puro!")

    return (
        f"^XA\n"
        f"^FO0,0\n"
        f"^GFA,{total_bytes},{total_bytes},{bytes_per_row},{zpl_hex}\n"
        f"^FS\n"
        f"^XZ\n"
    )


def _worker(args: tuple) -> tuple:
    """Worker para ThreadPoolExecutor: (page_idx, pdf_bytes, target_w, target_h) → (idx, zpl)"""
    page_idx, pdf_bytes, target_w, target_h = args
    img = renderizar_pagina(pdf_bytes, page_idx, target_w, target_h)
    zpl = imagem_para_zpl(img)
    return (page_idx, zpl)


def pdf_para_zpl(pdf_bytes: bytes, width_inches: float, height_inches: float) -> str:
    target_w = int(width_inches * DPI)
    target_h = int(height_inches * DPI)

    doc_check = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc_check)
    doc_check.close()

    logger.info(f"Iniciando: {n_pages} págs | {width_inches}x{height_inches}pol | target={target_w}x{target_h}px | workers={MAX_WORKERS}")

    args_list = [(i, pdf_bytes, target_w, target_h) for i in range(n_pages)]
    resultados = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_worker, args): args[0] for args in args_list}
        for future in as_completed(futures):
            try:
                idx, zpl = future.result()
                resultados[idx] = zpl
                done = len(resultados)
                if done % 20 == 0 or done == n_pages:
                    logger.info(f"  Progresso: {done}/{n_pages}")
            except Exception as e:
                idx = futures[future]
                logger.error(f"  Erro página {idx}: {e}")
                raise RuntimeError(f"Falha na página {idx}: {e}")

    zpl_final = "".join(resultados[i] for i in range(n_pages))

    n_xza = zpl_final.count("^XA")
    n_xzz = zpl_final.count("^XZ")
    logger.info(f"Concluído: {n_pages} págs | ^XA={n_xza} | ^XZ={n_xzz} | {len(zpl_final)/1024/1024:.1f} MB")

    if n_xza != n_pages or n_xzz != n_pages:
        raise RuntimeError(f"ZPL inválido: esperado {n_pages} pares ^XA/^XZ, encontrado {n_xza}/{n_xzz}")

    return zpl_final


# ──────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "4.0.0", "workers": MAX_WORKERS, "dpi": DPI}


@app.post("/convert", response_class=PlainTextResponse)
async def convert(
    file: UploadFile = File(...),
    width_inches: float = Form(7.0),
    height_inches: float = Form(5.0),
):
    """Recebe PDF, retorna ZPL completo (todas as páginas, 1 arquivo)."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Apenas arquivos .pdf são aceitos")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Arquivo vazio")
    logger.info(f"Recebido: {file.filename} | {len(pdf_bytes)/1024/1024:.1f} MB | {width_inches}x{height_inches}pol")
    try:
        zpl_content = pdf_para_zpl(pdf_bytes, width_inches, height_inches)
    except Exception as e:
        logger.error(f"Falha: {e}")
        raise HTTPException(500, str(e))
    return PlainTextResponse(
        content=zpl_content,
        headers={
            "X-Labels-Count": str(zpl_content.count("^XA")),
            "X-Size-MB": f"{len(zpl_content)/1024/1024:.2f}",
        },
    )


@app.post("/debug/render")
async def debug_render(
    file: UploadFile = File(...),
    width_inches: float = Form(7.0),
    height_inches: float = Form(5.0),
    page: int = Form(0),
):
    """
    DEBUG: retorna o PNG renderizado de uma página específica.
    Use no browser / Postman para ver o que o fitz está gerando antes de converter para ZPL.
    """
    pdf_bytes = await file.read()
    target_w = int(width_inches * DPI)
    target_h = int(height_inches * DPI)

    img = renderizar_pagina(pdf_bytes, page, target_w, target_h)

    pixels = list(img.getdata())
    stats = {
        "page": page,
        "rendered_size": img.size,
        "target_size": (target_w, target_h),
        "pixel_min": min(pixels),
        "pixel_max": max(pixels),
        "pixel_mean": round(sum(pixels) / len(pixels), 1),
        "width_inches": width_inches,
        "height_inches": height_inches,
        "dpi": DPI,
    }
    logger.info(f"Debug render: {stats}")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Adiciona estatísticas no header para facilitar debug
    return Response(
        content=buf.read(),
        media_type="image/png",
        headers={k: str(v) for k, v in stats.items()},
    )


@app.post("/convert/info")
async def convert_info(file: UploadFile = File(...)):
    """Info do PDF sem converter."""
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
