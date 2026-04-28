import io
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from PIL import Image

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF to ZPL Converter", version="4.3.0")

# Configurações via variáveis de ambiente
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
DPI = int(os.getenv("DPI", 203))

def renderizar_pagina(pdf_bytes: bytes, page_idx: int, target_w: int, target_h: int) -> Image.Image:
    """Abre o PDF e renderiza a página garantindo que o conteúdo seja capturado em fundo branco."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_idx]

    # Define o zoom baseado no DPI desejado
    zoom = DPI / 72
    mat = fitz.Matrix(zoom, zoom)
    
    # Renderiza em RGB com anotações (annots=True ajuda a pegar códigos de barras e campos preenchidos)
    pix = page.get_pixmap(matrix=mat, alpha=False, annots=True)
    
    # Cria imagem PIL a partir dos samples RGB
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    # Converte para escala de cinza (L) para processamento de bits
    img = img.convert("L")
    
    # Redimensiona para o tamanho exato da etiqueta se necessário
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    return img

def imagem_para_zpl(img: Image.Image) -> str:
    """Converte imagem grayscale para comando ZPL ^GFA (Gráficos binários)."""
    # Aplica um threshold (limiar) para converter em 1-bit (Preto e Branco puro)
    # Sem dithering para manter a precisão de códigos de barras
    img_1bit = img.point(lambda x: 0 if x < 128 else 255, mode='1')

    width, height = img_1bit.size
    bytes_per_row = (width + 7) // 8

    hex_data = []
    pixels = img_1bit.load()
    
    # Percorre a imagem e converte pixels para bytes hexadecimais
    for y in range(height):
        for x in range(0, bytes_per_row * 8, 8):
            byte = 0
            for bit in range(8):
                if x + bit < width:
                    # No ZPL, o bit 1 representa cor (preto), no PIL '1' o pixel preto é 0
                    if pixels[x + bit, y] == 0:
                        byte |= (1 << (7 - bit))
            hex_data.append(f"{byte:02X}")

    zpl_hex = "".join(hex_data)
    total_bytes = len(zpl_hex) // 2

    # Retorna o frame ZPL completo da página
    return f"^XA^FO0,0^GFA,{total_bytes},{total_bytes},{bytes_per_row},{zpl_hex}^FS^XZ\n"

def _worker(args: tuple) -> tuple:
    """Função auxiliar para processamento paralelo de páginas."""
    page_idx, pdf_bytes, target_w, target_h = args
    try:
        img = renderizar_pagina(pdf_bytes, page_idx, target_w, target_h)
        zpl = imagem_para_zpl(img)
        return (page_idx, zpl)
    except Exception as e:
        logger.error(f"Erro no processamento da página {page_idx}: {e}")
        return (page_idx, "")

@app.post("/convert", response_class=PlainTextResponse)
async def convert(
    file: UploadFile = File(...),
    width_cm: float = Form(None),
    height_cm: float = Form(None),
    width_inches: float = Form(None),
    height_inches: float = Form(None),
):
    """Endpoint principal de conversão."""
    # Lógica de prioridade de medidas (CM > Inches > Default)
    w_in = width_cm / 2.54 if width_cm else (width_inches or 4.0)
    h_in = height_cm / 2.54 if height_cm else (height_inches or 6.0)

    target_w = int(w_in * DPI)
    target_h = int(h_in * DPI)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Por favor, envie um arquivo PDF.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Arquivo PDF vazio.")

    try:
        # Verifica número de páginas
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            n_pages = len(doc)

        logger.info(f"Iniciando conversão de {n_pages} páginas ({w_in}x{h_in} pol)")

        # Prepara lista de tarefas para as threads
        args_list = [(i, pdf_bytes, target_w, target_h) for i in range(n_pages)]
        results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_worker, arg): arg[0] for arg in args_list}
            for future in as_completed(futures):
                results.append(future.result())

        # Ordena os resultados para garantir que as páginas fiquem na sequência correta
        results.sort(key=lambda x: x[0])
        zpl_final = "".join([r[1] for r in results])
        
        return zpl_final

    except Exception as e:
        logger.error(f"Erro crítico: {e}")
        raise HTTPException(500, f"Erro interno: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "version": "4.3.0"}
