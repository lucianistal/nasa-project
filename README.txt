Instrucciones rápidas:

1) Instalar dependencias (PowerShell):

```powershell
python -m pip install -r requirements.txt
```

2) Estructura:
- `anomaly/data.py`: carga y preprocesado
- `anomaly/models.py`: implementaciones de métodos
- `anomaly/train_eval.py`: orquestador para entrenar y detectar
- `NASA.ipynb`: Jupyter notebook
- `Laboratorio_Unidad_III_Informe.docx`: informe final sobre la tarea

3) Ejecución (lo importante):

```powershell
python -c "from anomaly.train_eval import evaluate_methods; evaluate_methods('.', 'FD001')"

```
