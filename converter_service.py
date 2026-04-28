"""
PDF → ZPL Converter Service v4.2
- Conversão automática CM -> Inches
- Otimização de limpeza de memória
- Renderização de anotações (annots=True)
- Sem dithering para maior nitidez em etiquetas térmicas
"""

import io
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF to ZPL Converter", version="4.2.0")

MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
DPI = int(os.getenv("DPI", 203))

def renderizar_pagina(pdf_bytes: bytes, page_idx: int, target_w: int, target_h: int) -> Image.Image:
    """Renderiza a página garantindo que elementos e anotações sejam visíveis."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_idx]

    # Matrix de escala baseada no DPI (72 pontos por polegada é o padrão PDF)
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    
    # annots=True captura carimbos e assinaturas que às vezes somem
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False, annots=True)

    img = Image.frombytes("L", (pix.width, pix.height), bytes(pix.samples))
    doc.close()

    # Redimensionamento final para garantir o encaixe exato na etiqueta
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    return img

def imagem_para_zpl(img: Image.Image) -> str:
    """Converte imagem para bloco ZPL otimizado para etiquetas térmicas."""
    # Convertemos para 1-bit (Preto e Branco puro) sem pontilhado
    img_1bit = img.convert("1", dither=None)

    width, height = img_1bit.size
    bytes_per_row = (width + 7) // 8

    hex_rows = []
    pixels = img_1bit.load()
    
    for y in range(height):
        row_bytes = []
        for x in range(0, width, 8):
            byte_val = 0
            for bit in range(8):
                if x + bit < width:
                    # No ZPL: 1=Preto (tinta), 0=Branco. No PIL "1": 0=Preto.
                    if pixels[x + bit, y] == 0:
                        byte_val |= (1 << (7 - bit))
            row_bytes.append(f"{byte_val:02X}")
        hex_rows.append("".join(row_bytes))

    zpl_hex = "".join(hex_rows)
    total_bytes = len(zpl_hex) // 2

    # Retorno em uma única linha compacta para evitar erros de buffer
    return f"^XA^FO0,0^GFA,{total_bytes},{total_bytes},{bytes_per_row},{zpl_hex}^FS^XZ"

def _worker(args: tuple) -> tuple:
    page_idx, pdf_bytes, target_w, target_h = args
    try:
        img = renderizar_pagina(pdf_bytes, page_idx, target_w, target_h)
        zpl = imagem_para_zpl(img)
        return (page_idx, zpl)
    except Exception as e:
        logger.error(f"Falha na página {page_idx}: {e}")
        raise

@app.post("/convert", response_class=PlainTextResponse)
async def convert(
    file: UploadFile = File(...),
    width_cm: float = Form(None),
    height_cm: float = Form(None),
    width_inches: float = Form(None),
    height_inches: float = Form(None),
):
    """
    Suporta entrada em CM ou Inches. Prioriza CM se ambos forem enviados.
    """
    # Lógica de conversão de medidas
    final_w_in = width_inches or 7.0
    final_h_in = height_inches or 5.0

    if width_cm:
        final_w_in = width_cm / 2.54
    if height_cm:
        final_h_in = height_cm / 2.54

    target_w = int(final_w_in * DPI)
    target_h = int(final_h_in * DPI)

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Arquivo vazio")

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            n_pages = len(doc)

        logger.info(f"Convertendo {n_pages} págs para {final_w_in:.2f}x{final_h_in:.2f} pol ({target_w}x{target_h}px)")

        args_list = [(i, pdf_bytes, target_w, target_h) for i in range(n_pages)]
        resultados = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_worker, args): args[0] for args in args_list}
            for future in as_completed(futures):
                idx, zpl = future.result()
                resultados[idx] = zpl

        zpl_final = "".join(resultados[i] for i in range(n_pages))
        return PlainTextResponse(content=zpl_final)

    except Exception as e:
        logger.error(f"Erro geral: {e}")
        raise HTTPException(500, str(e))

@app.get("/health")
def health():
    return {"status": "ready", "dpi": DPI, "workers": MAX_WORKERS}
