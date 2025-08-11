# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import base64
import json
import uuid
import os
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
import sqlite3
from contextlib import contextmanager

app = FastAPI(title="Report Generator API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class ChartConfig(BaseModel):
    chart_type: str
    x_column: str
    y_column: str
    title: Optional[str] = None


class ReportRequest(BaseModel):
    name: str
    chart_config: ChartConfig
    data: List[dict]


# Database setup
def init_db():
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS reports
                   (
                       id
                       TEXT
                       PRIMARY
                       KEY,
                       name
                       TEXT
                       NOT
                       NULL,
                       created_at
                       TEXT
                       NOT
                       NULL,
                       chart_type
                       TEXT
                       NOT
                       NULL,
                       x_column
                       TEXT
                       NOT
                       NULL,
                       y_column
                       TEXT
                       NOT
                       NULL,
                       title
                       TEXT,
                       file_path
                       TEXT,
                       data_json
                       TEXT
                   )
                   ''')
    conn.commit()
    conn.close()


@contextmanager
def get_db():
    conn = sqlite3.connect('reports.db')
    try:
        yield conn
    finally:
        conn.close()


# Utility Classes
class DataProcessor:
    @staticmethod
    def process_csv(file_content: bytes) -> dict:
        try:
            df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
            df.columns = df.columns.str.strip()

            info = {
                'rows': len(df),
                'columns': list(df.columns),
                'data_types': df.dtypes.astype(str).to_dict(),
                'sample_data': df.head(5).to_dict('records'),
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist()
            }

            return {
                'status': 'success',
                'data': df.to_dict('records'),
                'info': info
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    @staticmethod
    def generate_chart_data(data: List[dict], x_col: str, y_col: str) -> dict:
        try:
            df = pd.DataFrame(data)

            # Try to convert y_col to numeric if possible
            try:
                df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                chart_data = df.groupby(x_col)[y_col].sum().reset_index()
            except:
                # If conversion fails, count occurrences
                chart_data = df.groupby(x_col).size().reset_index(name=y_col)

            chart_data.columns = ['name', 'value']
            result = chart_data.to_dict('records')

            # Statistics
            values = [item['value'] for item in result]
            stats = {
                'total_points': len(result),
                'max_value': float(max(values)) if values else 0,
                'min_value': float(min(values)) if values else 0,
                'avg_value': float(sum(values) / len(values)) if values else 0,
                'sum_value': float(sum(values)) if values else 0
            }

            return {
                'chart_data': result,
                'statistics': stats
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao processar dados do gráfico: {str(e)}")


class ChartGenerator:
    @staticmethod
    def create_matplotlib_chart(data: List[dict], chart_type: str, title: str = None) -> str:
        try:
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(10, 6))

            names = [str(item['name']) for item in data]
            values = [float(item['value']) for item in data]

            if chart_type == 'bar':
                bars = ax.bar(names, values, color='steelblue', alpha=0.7)
                ax.set_xlabel('Categorias')
                ax.set_ylabel('Valores')

                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{int(height)}', ha='center', va='bottom')

            elif chart_type == 'line':
                ax.plot(names, values, marker='o', linewidth=2, markersize=6, color='steelblue')
                ax.set_xlabel('Categorias')
                ax.set_ylabel('Valores')
                ax.grid(True, alpha=0.3)

            elif chart_type == 'pie':
                colors_list = plt.cm.Set3(np.linspace(0, 1, len(data)))
                wedges, texts, autotexts = ax.pie(values, labels=names, autopct='%1.1f%%',
                                                  colors=colors_list, startangle=90)
                ax.axis('equal')

            if title:
                ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

            if chart_type != 'pie':
                plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64
        except Exception as e:
            print(f"Erro ao gerar gráfico: {e}")
            raise


class PDFGenerator:
    @staticmethod
    def create_report_pdf(report_data: dict) -> str:
        try:
            filename = f"report_{uuid.uuid4().hex[:8]}.pdf"
            os.makedirs("reports", exist_ok=True)
            filepath = f"reports/{filename}"

            doc = SimpleDocTemplate(filepath, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.HexColor('#1f2937')
            )
            story.append(Paragraph(report_data['name'], title_style))
            story.append(Spacer(1, 12))

            # Report Info
            info_data = [
                ['Gerado em:', datetime.now().strftime('%d/%m/%Y %H:%M:%S')],
                ['Tipo de Gráfico:', report_data['chart_config']['chart_type'].title()],
                ['Eixo X:', report_data['chart_config']['x_column']],
                ['Eixo Y:', report_data['chart_config']['y_column']],
            ]

            info_table = Table(info_data, colWidths=[2 * inch, 4 * inch])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.grey),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))

            story.append(info_table)
            story.append(Spacer(1, 20))

            # Chart Image
            # Dentro de PDFGenerator.create_report_pdf
            if 'chart_image' in report_data:
                try:
                    image_data = base64.b64decode(report_data['chart_image'])
                    temp_image_path = f"temp_chart_{uuid.uuid4().hex[:8]}.png"

                    with open(temp_image_path, 'wb') as f:
                        f.write(image_data)

                    story.append(Image(temp_image_path, width=6 * inch, height=3.6 * inch))
                    story.append(Spacer(1, 20))

                except Exception as e:
                    print(f"Erro ao processar imagem: {e}")

            doc.build(story)

            # Só remove aqui
            if 'chart_image' in report_data and os.path.exists(temp_image_path):
                os.remove(temp_image_path)

            # Statistics
            if 'statistics' in report_data:
                stats = report_data['statistics']
                stats_data = [
                    ['Estatística', 'Valor'],
                    ['Total de Pontos', str(stats['total_points'])],
                    ['Valor Máximo', f"{stats['max_value']:,.2f}"],
                    ['Valor Mínimo', f"{stats['min_value']:,.2f}"],
                    ['Valor Médio', f"{stats['avg_value']:,.2f}"],
                    ['Soma dos Valores', f"{stats['sum_value']:,.2f}"],
                ]

                stats_table = Table(stats_data, colWidths=[2 * inch, 2 * inch])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))

                story.append(Paragraph("Resumo Estatístico", styles['Heading2']))
                story.append(Spacer(1, 12))
                story.append(stats_table)

            doc.build(story)
            return filepath
        except Exception as e:
            print(f"Erro ao gerar PDF: {e}")
            raise


# Initialize database
init_db()


# API Endpoints
@app.get("/")
async def root():
    return {"message": "API Gerador de Relatórios", "version": "1.0.0"}


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Apenas arquivos CSV são permitidos")

    content = await file.read()
    result = DataProcessor.process_csv(content)

    if result['status'] == 'error':
        raise HTTPException(status_code=400, detail=result['message'])

    return result


@app.post("/generate-chart")
async def generate_chart(request: dict):
    try:
        data = request['data']
        x_col = request['x_column']
        y_col = request['y_column']
        chart_type = request['chart_type']
        title = request.get('title', f'{y_col} por {x_col}')

        chart_result = DataProcessor.generate_chart_data(data, x_col, y_col)

        chart_image = ChartGenerator.create_matplotlib_chart(
            chart_result['chart_data'], chart_type, title
        )

        return {
            'chart_data': chart_result['chart_data'],
            'statistics': chart_result['statistics'],
            'chart_image': chart_image
        }
    except Exception as e:
        print(f"Erro ao gerar gráfico: {e}")
        raise HTTPException(status_code=400, detail=f"Erro ao gerar gráfico: {str(e)}")


@app.post("/save-report")
async def save_report(request: ReportRequest):
    try:
        report_id = str(uuid.uuid4())

        # Gerar dados do gráfico
        chart_result = DataProcessor.generate_chart_data(
            request.data,
            request.chart_config.x_column,
            request.chart_config.y_column
        )

        # Gerar imagem do gráfico
        chart_image = ChartGenerator.create_matplotlib_chart(
            chart_result['chart_data'],
            request.chart_config.chart_type,
            request.chart_config.title or request.name
        )

        # Preparar dados para o PDF - CORRIGIDO: usando model_dump() ao invés de dict()
        pdf_data = {
            'name': request.name,
            'chart_config': request.chart_config.model_dump(),  # CORRIGIDO AQUI
            'chart_image': chart_image,
            'statistics': chart_result['statistics']
        }

        # Gerar PDF
        pdf_path = PDFGenerator.create_report_pdf(pdf_data)

        # Salvar no banco de dados
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                           INSERT INTO reports (id, name, created_at, chart_type, x_column, y_column, title, file_path,
                                                data_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ''', (
                               report_id,
                               request.name,
                               datetime.now().isoformat(),
                               request.chart_config.chart_type,
                               request.chart_config.x_column,
                               request.chart_config.y_column,
                               request.chart_config.title,
                               pdf_path,
                               json.dumps(request.data)
                           ))
            conn.commit()

        return {
            'id': report_id,
            'message': 'Relatório salvo com sucesso',
            'pdf_path': pdf_path
        }
    except Exception as e:
        print(f"Erro ao salvar relatório: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao salvar relatório: {str(e)}")


@app.get("/reports")
async def get_reports():
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, name, created_at, chart_type, x_column, y_column, title FROM reports ORDER BY created_at DESC')
            rows = cursor.fetchall()

            reports = []
            for row in rows:
                reports.append({
                    'id': row[0],
                    'name': row[1],
                    'created_at': row[2],
                    'chart_type': row[3],
                    'x_column': row[4],
                    'y_column': row[5],
                    'title': row[6]
                })

            return {'reports': reports}
    except Exception as e:
        print(f"Erro ao carregar relatórios: {e}")
        raise HTTPException(status_code=500, detail="Erro ao carregar relatórios")


@app.get("/download-report/{report_id}")
async def download_report(report_id: str):
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT file_path, name FROM reports WHERE id = ?', (report_id,))
            row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Relatório não encontrado")

            if not os.path.exists(row[0]):
                raise HTTPException(status_code=404, detail="Arquivo do relatório não encontrado")

            return FileResponse(
                path=row[0],
                filename=f"{row[1]}.pdf",
                media_type='application/pdf'
            )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Erro ao fazer download: {e}")
        raise HTTPException(status_code=500, detail="Erro ao fazer download do relatório")


@app.delete("/reports/{report_id}")
async def delete_report(report_id: str):
    try:
        with get_db() as conn:
            cursor = conn.cursor()

            # Buscar o caminho do arquivo
            cursor.execute('SELECT file_path FROM reports WHERE id = ?', (report_id,))
            row = cursor.fetchone()

            # Remover arquivo físico se existir
            if row and row[0] and os.path.exists(row[0]):
                try:
                    os.remove(row[0])
                except Exception as e:
                    print(f"Erro ao remover arquivo: {e}")

            # Remover do banco de dados
            cursor.execute('DELETE FROM reports WHERE id = ?', (report_id,))
            conn.commit()

            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Relatório não encontrado")

            return {'message': 'Relatório deletado com sucesso'}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Erro ao deletar relatório: {e}")
        raise HTTPException(status_code=500, detail="Erro ao deletar relatório")


if __name__ == "__main__":
    import uvicorn

    print("🚀 Iniciando servidor da API...")
    print("📊 API Gerador de Relatórios v1.0.0")
    print("🌐 Servidor rodando em: http://localhost:8000")
    print("📚 Documentação disponível em: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)