import os
import sys
import shutil
import logging
from datetime import datetime
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Garantir que o servidor encontre os módulos do projeto
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_training'))

# Imports dos scripts refatorados
from main import run_processing
from train_model import run_training

# Configuração de Logs para monitoramento interno
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AgroVisionAPI")

app = FastAPI(title="AgroVision API", description="Servidor para coleta e treinamento remoto (Suporte ONNX)")

# --- MODELOS DE DADOS (Pydantic) ---
class StatusResponse(BaseModel):
    milho: int
    erva_daninha: int
    total: int
    balanceamento: str

class TrainResponse(BaseModel):
    message: str
    csv_file: str
    model_pkl: str
    model_onnx: str
    samples: int

class ModelInfo(BaseModel):
    filename: str
    format: str
    size_bytes: int
    created_at: str

# --- UTILITÁRIOS ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT_DIR, "data/raw")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data/output")

def get_next_sequence(date_str: str) -> str:
    """Verifica qual a próxima sequência (seq01, seq02...) para a data atual."""
    # Prioriza verificar arquivos .onnx para definir a sequência
    files = [f for f in os.listdir(OUTPUT_DIR) if date_str in f and (f.endswith(".onnx") or f.endswith(".pkl"))]
    if not files:
        return "seq01"
    
    sequences = []
    for f in files:
        try:
            # modelo_YYYYMMDD_seqXX.ext -> extrai o XX
            if "_seq" in f:
                part = f.split("_seq")[-1].split(".")[0]
                sequences.append(int(part))
        except: continue
    
    next_num = max(sequences) + 1 if sequences else 1
    return f"seq{next_num:02d}"

# --- ENDPOINTS ---

@app.post("/upload", tags=["Coleta"])
async def upload_image(classe: str, file: UploadFile = File(...)):
    """Envia uma foto para a pasta raw correta com timestamp."""
    if classe not in ["milho", "erva_daninha"]:
        logger.error(f"Tentativa de upload para classe inválida: {classe}")
        raise HTTPException(status_code=400, detail="Classe deve ser 'milho' ou 'erva_daninha'")
    
    target_folder = os.path.join(RAW_DIR, classe)
    os.makedirs(target_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"coleta_{timestamp}_{file.filename}"
    file_path = os.path.join(target_folder, filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Foto salva: {filename} na pasta {classe}")
        return {"message": "Upload concluído", "path": filename}
    except Exception as e:
        logger.error(f"Erro no upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao salvar arquivo")

@app.get("/status", response_model=StatusResponse, tags=["Monitoramento"])
async def get_status():
    """Retorna a contagem atual de imagens para balanço."""
    counts = {}
    total = 0
    for cls in ["milho", "erva_daninha"]:
        p = os.path.join(RAW_DIR, cls)
        count = len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]) if os.path.exists(p) else 0
        counts[cls] = count
        total += count
    
    if total == 0:
        bal = "vazio"
    else:
        diff = abs(counts["milho"] - counts["erva_daninha"])
        bal = "equilibrado" if diff < (total * 0.20) else "desbalanceado"

    logger.info(f"Status solicitado: {counts}")
    return {
        "milho": counts["milho"],
        "erva_daninha": counts["erva_daninha"],
        "total": total,
        "balanceamento": bal
    }

@app.post("/train", response_model=TrainResponse, tags=["Treinamento"])
async def train_model():
    """Executa processamento e treinamento gerando PKL e ONNX."""
    date_str = datetime.now().strftime("%Y%m%d")
    seq = get_next_sequence(date_str)
    
    version_label = f"{date_str}_{seq}"
    csv_filename = f"dataset_{version_label}.csv"
    model_filename = f"modelo_{version_label}.pkl"
    onnx_filename = f"modelo_{version_label}.onnx"
    
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    model_path = os.path.join(OUTPUT_DIR, model_filename)
    
    logger.info(f"Iniciando ciclo de treinamento (ONNX): {version_label}")
    
    try:
        # 1. Processamento (ExG + Augmentation)
        _, samples = run_processing(output_csv_path=csv_path)
        
        if samples == 0:
            raise HTTPException(status_code=400, detail="Nenhuma imagem encontrada para treinar.")
            
        # 2. Treinamento (Random Forest -> PKL & ONNX)
        final_model_path = run_training(csv_path=csv_path, model_output_path=model_path)
        
        if not final_model_path:
            raise Exception("Falha ao gerar arquivo de modelo.")
            
        logger.info(f"Treinamento finalizado com sucesso: {version_label}")
        
        return {
            "message": "Treinamento concluído com sucesso",
            "csv_file": csv_filename,
            "model_pkl": model_filename,
            "model_onnx": onnx_filename,
            "samples": samples
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Erro durante o treinamento:\n{error_details}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=List[ModelInfo], tags=["Modelos"])
async def list_models():
    """Lista todos os modelos (.onnx e .pkl) disponíveis no servidor."""
    try:
        files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith((".pkl", ".onnx"))]
        files.sort(reverse=True)
        
        models = []
        for f in files:
            path = os.path.join(OUTPUT_DIR, f)
            stats = os.stat(path)
            models.append({
                "filename": f,
                "format": "ONNX (Android)" if f.endswith(".onnx") else "PKL (Python)",
                "size_bytes": stats.st_size,
                "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat()
            })
        return models
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao listar arquivos")

@app.get("/models/download/{filename}", tags=["Modelos"])
async def download_model(filename: str):
    """Faz o download de um arquivo de modelo específico (.onnx ou .pkl)."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        logger.warning(f"Tentativa de download de arquivo inexistente: {filename}")
        raise HTTPException(status_code=404, detail="Arquivo de modelo não encontrado")
    
    logger.info(f"Enviando arquivo: {filename}")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando servidor AgroVision (Suporte ONNX)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
