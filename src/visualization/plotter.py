"""
Sistema de visualização para análise de simulações
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import sys

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.models import HTCEvent, InterscsimulatorEvent, ComparisonResult
from src.metrics.calculator import BasicMetrics, TrafficMetrics, LinkMetrics

# Tentar importar configurações
try:
    from config.settings import VISUALIZATION_CONFIG, OUTPUT_CONFIG
except ImportError:
    # Configurações padrão se não encontrar o arquivo
    VISUALIZATION_CONFIG = {
        'figure_size': (12, 8),
        'dpi': 300,
        'color_palette': 'viridis',
        'heatmap_resolution': 100
    }
    OUTPUT_CONFIG = {
        'base_dir': './output',
        'reports_dir': './output/reports',
        'plots_dir': './output/plots',
        'comparison_dir': './output/comparison'
    }


class SimulationVisualizer:
    """Visualizador para análise de simulações"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(OUTPUT_CONFIG['plots_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurações de estilo
        self.figure_size = VISUALIZATION_CONFIG['figure_size']
        self.dpi = VISUALIZATION_CONFIG['dpi']
        self.color_palette = VISUALIZATION_CONFIG['color_palette']
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        sns.set_palette(self.color_palette)
        
        self.logger = logging.getLogger(__name__)
    
    def plot_basic_metrics_comparison(self, 
                                    htc_metrics: BasicMetrics, 
                                    ref_metrics: BasicMetrics,
                                    save_path: Optional[str] = None) -> str:
        """Gráfico de comparação de métricas básicas"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Comparação de Métricas Básicas', fontsize=16)
        
        metrics_data = {
            'HTC': [
                htc_metrics.total_vehicles,
                htc_metrics.total_distance,
                htc_metrics.average_speed,
                htc_metrics.average_travel_time,
                htc_metrics.simulation_duration,
                htc_metrics.throughput
            ],
            'Interscsimulator': [
                ref_metrics.total_vehicles,
                ref_metrics.total_distance,
                ref_metrics.average_speed,
                ref_metrics.average_travel_time,
                ref_metrics.simulation_duration,
                ref_metrics.throughput
            ]
        }
        
        metric_names = [
            'Total Vehicles', 'Total Distance (m)', 'Avg Speed (m/s)',
            'Avg Travel Time (s)', 'Duration (s)', 'Throughput (veh/s)'
        ]
        
        for i, (ax, metric_name) in enumerate(zip(axes.flat, metric_names)):
            values = [metrics_data['HTC'][i], metrics_data['Interscsimulator'][i]]
            labels = ['HTC', 'Interscsimulator']
            
            bars = ax.bar(labels, values, color=['#1f77b4', '#ff7f0e'])
            ax.set_title(metric_name)
            ax.set_ylabel('Value')
            
            # Adicionar valores nas barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'basic_metrics_comparison.png'
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Gráfico de métricas básicas salvo em: {save_path}")
        return str(save_path)
    
    def plot_speed_distribution(self, 
                              htc_events: List[HTCEvent], 
                              ref_events: List[InterscsimulatorEvent],
                              save_path: Optional[str] = None) -> str:
        """Gráfico de distribuição de velocidades"""
        
        # Extrair velocidades
        htc_speeds = []
        ref_speeds = []
        
        for event in htc_events:
            if event.event_type == 'enter_link' and 'calculated_speed' in event.data:
                htc_speeds.append(event.data['calculated_speed'])
        
        for event in ref_events:
            if event.event_type == 'enter_link' and 'calculated_speed' in event.attributes:
                ref_speeds.append(float(event.attributes['calculated_speed']))
        
        # Criar gráfico
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogramas
        ax1.hist(htc_speeds, bins=50, alpha=0.7, label='HTC', density=True)
        ax1.hist(ref_speeds, bins=50, alpha=0.7, label='Interscsimulator', density=True)
        ax1.set_xlabel('Velocidade (m/s)')
        ax1.set_ylabel('Densidade')
        ax1.set_title('Distribuição de Velocidades')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plots
        box_data = [htc_speeds, ref_speeds]
        ax2.boxplot(box_data, labels=['HTC', 'Interscsimulator'])
        ax2.set_ylabel('Velocidade (m/s)')
        ax2.set_title('Box Plot de Velocidades')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'speed_distribution.png'
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Gráfico de distribuição de velocidades salvo em: {save_path}")
        return str(save_path)
    
    def plot_density_heatmap(self, 
                           density_data: pd.DataFrame,
                           title: str = "Densidade de Links",
                           save_path: Optional[str] = None) -> str:
        """Gráfico de mapa de calor de densidade"""
        
        # Criar pivot table para heatmap
        if 'time_bin' not in density_data.columns:
            # Criar bins de tempo se não existirem
            density_data['time_bin'] = pd.cut(density_data['timestamp'], bins=20)
        
        heatmap_data = density_data.groupby(['link_id', 'time_bin'])['density'].mean().unstack(fill_value=0)
        
        # Criar heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(heatmap_data, 
                   cmap='YlOrRd', 
                   cbar_kws={'label': 'Densidade Média'},
                   xticklabels=False)  # Remover labels do eixo x por clareza
        
        plt.title(title)
        plt.xlabel('Período de Tempo')
        plt.ylabel('Link ID')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f'density_heatmap_{title.lower().replace(" ", "_")}.png'
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Mapa de calor de densidade salvo em: {save_path}")
        return str(save_path)
    
    def plot_temporal_metrics(self, 
                            temporal_data: pd.DataFrame,
                            title: str = "Métricas Temporais",
                            save_path: Optional[str] = None) -> str:
        """Gráfico de métricas ao longo do tempo"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Métricas a plotar
        metrics = ['unique_vehicles', 'avg_speed', 'avg_density', 'total_events']
        titles = ['Veículos Únicos', 'Velocidade Média', 'Densidade Média', 'Total de Eventos']
        
        for ax, metric, subtitle in zip(axes.flat, metrics, titles):
            if metric in temporal_data.columns:
                ax.plot(temporal_data['time'], temporal_data[metric], marker='o', markersize=4)
                ax.set_title(subtitle)
                ax.set_xlabel('Tempo')
                ax.set_ylabel(subtitle)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f'temporal_metrics_{title.lower().replace(" ", "_")}.png'
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Gráfico de métricas temporais salvo em: {save_path}")
        return str(save_path)
    
    def plot_link_comparison(self, 
                           comparison_data: pd.DataFrame,
                           save_path: Optional[str] = None) -> str:
        """Gráfico de comparação por link"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparação por Link', fontsize=16)
        
        # Scatter plots de comparação
        metrics = [
            ('htc_density_mean', 'ref_density_mean', 'Densidade Média'),
            ('htc_calculated_speed_mean', 'ref_calculated_speed_mean', 'Velocidade Média'),
            ('htc_density_max', 'ref_density_max', 'Densidade Máxima'),
            ('density_diff', 'speed_diff', 'Diferenças (Densidade vs Velocidade)')
        ]
        
        for ax, (x_col, y_col, title) in zip(axes.flat, metrics):
            if x_col in comparison_data.columns and y_col in comparison_data.columns:
                ax.scatter(comparison_data[x_col], comparison_data[y_col], alpha=0.7)
                
                # Linha de igualdade para os primeiros 3 gráficos
                if title != 'Diferenças (Densidade vs Velocidade)':
                    max_val = max(comparison_data[x_col].max(), comparison_data[y_col].max())
                    min_val = min(comparison_data[x_col].min(), comparison_data[y_col].min())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Linha de Igualdade')
                    ax.legend()
                
                ax.set_xlabel(f'HTC - {title}')
                ax.set_ylabel(f'Interscsimulator - {title}')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'link_comparison.png'
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Gráfico de comparação por link salvo em: {save_path}")
        return str(save_path)
    
    def plot_vehicle_journey(self, 
                           htc_journey: List[Any], 
                           ref_journey: List[Any],
                           vehicle_id: str,
                           save_path: Optional[str] = None) -> str:
        """Gráfico da jornada de um veículo específico"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'Jornada do Veículo {vehicle_id}', fontsize=16)
        
        # Timeline de eventos
        htc_times = [event.timestamp for event in htc_journey]
        ref_times = [event.timestamp for event in ref_journey]
        
        htc_events = [event.event_type for event in htc_journey]
        ref_events = [event.event_type for event in ref_journey]
        
        # Mapear tipos de evento para números
        event_types = list(set(htc_events + ref_events))
        event_map = {event: i for i, event in enumerate(event_types)}
        
        htc_y = [event_map[event] for event in htc_events]
        ref_y = [event_map[event] for event in ref_events]
        
        ax1.scatter(htc_times, htc_y, label='HTC', alpha=0.7, s=50)
        ax1.scatter(ref_times, ref_y, label='Interscsimulator', alpha=0.7, s=50)
        ax1.set_xlabel('Tempo')
        ax1.set_ylabel('Tipo de Evento')
        ax1.set_yticks(list(event_map.values()))
        ax1.set_yticklabels(list(event_map.keys()))
        ax1.set_title('Timeline de Eventos')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Velocidades ao longo do tempo (se disponível)
        htc_speeds = []
        htc_speed_times = []
        ref_speeds = []
        ref_speed_times = []
        
        for event in htc_journey:
            if event.event_type == 'enter_link' and hasattr(event, 'data') and 'calculated_speed' in event.data:
                htc_speeds.append(event.data['calculated_speed'])
                htc_speed_times.append(event.timestamp)
        
        for event in ref_journey:
            if event.event_type == 'enter_link' and hasattr(event, 'attributes') and 'calculated_speed' in event.attributes:
                ref_speeds.append(float(event.attributes['calculated_speed']))
                ref_speed_times.append(event.timestamp)
        
        if htc_speeds or ref_speeds:
            ax2.plot(htc_speed_times, htc_speeds, 'o-', label='HTC', markersize=4)
            ax2.plot(ref_speed_times, ref_speeds, 's-', label='Interscsimulator', markersize=4)
            ax2.set_xlabel('Tempo')
            ax2.set_ylabel('Velocidade (m/s)')
            ax2.set_title('Velocidade ao Longo da Jornada')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Dados de velocidade não disponíveis', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f'vehicle_journey_{vehicle_id}.png'
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Gráfico de jornada do veículo salvo em: {save_path}")
        return str(save_path)
    
    def create_interactive_dashboard(self, 
                                   comparison_result: ComparisonResult,
                                   htc_temporal: pd.DataFrame,
                                   ref_temporal: pd.DataFrame,
                                   save_path: Optional[str] = None) -> str:
        """Cria dashboard interativo com Plotly"""
        
        # Criar subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Métricas Temporais - Veículos', 'Métricas Temporais - Velocidade',
                'Densidade Temporal', 'Comparação de Throughput',
                'Correlações', 'Resumo de Similaridade'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Gráfico 1: Veículos únicos ao longo do tempo
        if not htc_temporal.empty and 'unique_vehicles' in htc_temporal.columns:
            fig.add_trace(
                go.Scatter(x=htc_temporal['time'], y=htc_temporal['unique_vehicles'],
                          name='HTC', line=dict(color='blue')),
                row=1, col=1
            )
        
        if not ref_temporal.empty and 'unique_vehicles' in ref_temporal.columns:
            fig.add_trace(
                go.Scatter(x=ref_temporal['time'], y=ref_temporal['unique_vehicles'],
                          name='Interscsimulator', line=dict(color='red')),
                row=1, col=1
            )
        
        # Gráfico 2: Velocidade média
        if not htc_temporal.empty and 'avg_speed' in htc_temporal.columns:
            fig.add_trace(
                go.Scatter(x=htc_temporal['time'], y=htc_temporal['avg_speed'],
                          name='HTC Speed', line=dict(color='lightblue')),
                row=1, col=2
            )
        
        if not ref_temporal.empty and 'avg_speed' in ref_temporal.columns:
            fig.add_trace(
                go.Scatter(x=ref_temporal['time'], y=ref_temporal['avg_speed'],
                          name='Interscsimulator Speed', line=dict(color='lightcoral')),
                row=1, col=2
            )
        
        # Gráfico 3: Densidade
        if not htc_temporal.empty and 'avg_density' in htc_temporal.columns:
            fig.add_trace(
                go.Scatter(x=htc_temporal['time'], y=htc_temporal['avg_density'],
                          name='HTC Density', line=dict(color='green')),
                row=2, col=1
            )
        
        if not ref_temporal.empty and 'avg_density' in ref_temporal.columns:
            fig.add_trace(
                go.Scatter(x=ref_temporal['time'], y=ref_temporal['avg_density'],
                          name='Interscsimulator Density', line=dict(color='orange')),
                row=2, col=1
            )
        
        # Gráfico 4: Throughput comparison
        if not htc_temporal.empty and not ref_temporal.empty:
            htc_throughput = htc_temporal['unique_vehicles'].sum() / len(htc_temporal) if len(htc_temporal) > 0 else 0
            ref_throughput = ref_temporal['unique_vehicles'].sum() / len(ref_temporal) if len(ref_temporal) > 0 else 0
            
            fig.add_trace(
                go.Bar(x=['HTC', 'Interscsimulator'], y=[htc_throughput, ref_throughput],
                      name='Throughput'),
                row=2, col=2
            )
        
        # Gráfico 5: Correlações
        if comparison_result.correlation_metrics:
            corr_names = list(comparison_result.correlation_metrics.keys())
            corr_values = list(comparison_result.correlation_metrics.values())
            
            fig.add_trace(
                go.Bar(x=corr_names, y=corr_values, name='Correlações'),
                row=3, col=1
            )
        
        # Gráfico 6: Score de similaridade
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=comparison_result.similarity_score * 100,
                title={'text': "Similaridade (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 80], 'color': "yellow"},
                           {'range': [80, 100], 'color': "green"}
                       ],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=3, col=2
        )
        
        # Atualizar layout com descrições metodológicas
        methodology_text = """
        <b>Metodologia de Cálculo:</b><br>
        • Velocidade Média: Média aritmética de eventos 'enter_link'<br>
        • Throughput: Veículos únicos / duração da simulação<br>
        • Similaridade: Métrica composta (veículos + eventos + temporal)<br>
        • Janelas Temporais Adaptativas: Baseadas na granularidade específica de cada simulador
        """
        
        fig.update_layout(
            title={
                'text': "Dashboard de Comparação de Simulações<br><sub>Análise Comparativa entre HTC e Interscsimulator</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=True,
            height=1200,
            annotations=[
                dict(
                    text=methodology_text,
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    font=dict(size=10)
                )
            ]
        )
        
        # Adicionar anotações específicas para cada subplot
        fig.update_xaxes(title_text="Tipo de Evento", row=1, col=1)
        fig.update_yaxes(title_text="Contagem", row=1, col=1)
        
        fig.update_xaxes(title_text="Tempo", row=1, col=2)
        fig.update_yaxes(title_text="Velocidade Média (km/h)", row=1, col=2)
        
        fig.update_xaxes(title_text="Tempo", row=2, col=1)
        fig.update_yaxes(title_text="Densidade Média", row=2, col=1)
        
        fig.update_xaxes(title_text="Simulador", row=2, col=2)
        fig.update_yaxes(title_text="Throughput (veículos/tempo)", row=2, col=2)
        
        fig.update_xaxes(title_text="Métrica de Correlação", row=3, col=1)
        fig.update_yaxes(title_text="Valor da Correlação", row=3, col=1)
        
        if save_path is None:
            save_path = self.output_dir / 'interactive_dashboard.html'
        
        # Adicionar HTML customizado com mais detalhes metodológicos
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard Interativo - Comparação de Simulações</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .methodology {{ background-color: #f0f8ff; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007acc; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Dashboard Interativo - Comparação de Simulações</h1>
            
            <div class="methodology">
                <h3>🔬 Metodologia de Análise</h3>
                <p><strong>Processamento Adaptativo:</strong> O sistema detecta automaticamente o tipo de simulador e aplica janelas temporais otimizadas baseadas na granularidade dos dados:</p>
                <ul>
                    <li><strong>HTC:</strong> Janelas de max(1, duração/100) ticks para alta resolução temporal</li>
                    <li><strong>Interscsimulator:</strong> Janelas de max(30, duração/50) ticks para suavização</li>
                </ul>
                <p><strong>Normalização:</strong> IDs de veículos e links são normalizados para permitir comparação direta entre simuladores.</p>
            </div>
            
            <div class="warning">
                <h3>⚠️ Interpretação dos Resultados</h3>
                <p><strong>Correlações:</strong> |r| ≥ 0.8 (muito forte), 0.6-0.8 (forte), 0.4-0.6 (moderada), 0.2-0.4 (fraca), < 0.2 (muito fraca)</p>
                <p><strong>P-valores:</strong> < 0.05 indica diferença estatisticamente significativa</p>
                <p><strong>Similaridade:</strong> 0-1 onde 1 = simulações idênticas</p>
            </div>
            
            <div id="plotly-div">
                {self._get_plotly_div_content()}
            </div>
            
            <div class="methodology">
                <h3>📊 Descrição dos Gráficos</h3>
                <ul>
                    <li><strong>Eventos por Tipo:</strong> Distribuição de tipos de eventos para avaliar consistência comportamental</li>
                    <li><strong>Velocidade Temporal:</strong> Evolução da velocidade ao longo do tempo (janelas adaptativas)</li>
                    <li><strong>Densidade Temporal:</strong> Evolução da densidade de tráfego (carros por link)</li>
                    <li><strong>Throughput:</strong> Taxa de processamento de veículos por simulador</li>
                    <li><strong>Correlações:</strong> Força das relações lineares entre métricas dos simuladores</li>
                    <li><strong>Similaridade:</strong> Gauge indicando o nível geral de semelhança entre simulações</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        fig.write_html(save_path, include_plotlyjs='cdn')
        
        self.logger.info(f"Dashboard interativo salvo em: {save_path}")
        return str(save_path)
    
    def _get_plotly_div_content(self) -> str:
        """Retorna o conteúdo do div do Plotly para inserção no template"""
        return "<!-- Plotly chart will be inserted here -->"
    
    def _get_metric_descriptions(self) -> Dict[str, str]:
        """Retorna descrições detalhadas de como cada métrica é calculada"""
        return {
            # Métricas de Similaridade
            "similarity_score": """
            <strong>Score de Similaridade Geral:</strong><br>
            Métrica composta que avalia o quão similares são as duas simulações através de três componentes:<br>
            • <em>Similaridade de Veículos:</em> 1 - |N_htc - N_ref| / max(N_htc, N_ref)<br>
            • <em>Similaridade de Tipos de Eventos:</em> Comparação das proporções de cada tipo de evento<br>
            • <em>Similaridade Temporal:</em> 1 - |duração_htc - duração_ref| / max(duração_htc, duração_ref)<br>
            O score final é a média aritmética desses três componentes (0-1, onde 1 = idêntico).
            """,
            
            "reproducibility_score": """
            <strong>Score de Reprodutibilidade:</strong><br>
            Avalia se as simulações reproduzem os mesmos comportamentos através de:<br>
            • <em>Sobreposição de Veículos:</em> |veículos_comuns| / |veículos_totais|<br>
            • <em>Similaridade de Rotas:</em> Para veículos comuns, compara sequências de links visitados<br>
            O score final é a média ponderada desses componentes (0-1, onde 1 = perfeitamente reprodutível).
            """,
            
            # Correlações
            "pearson": """
            <strong>Correlação de Pearson:</strong><br>
            Mede a relação linear entre variáveis das duas simulações.<br>
            Fórmula: r = Σ[(x_i - x̄)(y_i - ȳ)] / √[Σ(x_i - x̄)²Σ(y_i - ȳ)²]<br>
            Valores: -1 (correlação negativa perfeita) a +1 (correlação positiva perfeita).
            """,
            
            "spearman": """
            <strong>Correlação de Spearman:</strong><br>
            Mede relações monotônicas (não necessariamente lineares) entre variáveis.<br>
            Baseada nos rankings dos dados ao invés dos valores absolutos.<br>
            Mais robusta a outliers que a correlação de Pearson.
            """,
            
            # Testes Estatísticos
            "speed_t_test": """
            <strong>Teste t para Velocidades:</strong><br>
            Testa se as médias de velocidade entre simulações são estatisticamente diferentes.<br>
            H₀: μ_htc = μ_ref (médias iguais)<br>
            H₁: μ_htc ≠ μ_ref (médias diferentes)<br>
            p < 0.05 indica diferença significativa.
            """,
            
            "speed_ks_test": """
            <strong>Teste Kolmogorov-Smirnov para Velocidades:</strong><br>
            Testa se as distribuições de velocidade são idênticas.<br>
            Compara as funções de distribuição cumulativa (CDF) das duas amostras.<br>
            Mais sensível a diferenças na forma da distribuição que o teste t.
            """,
            
            "density_mannwhitney": """
            <strong>Teste Mann-Whitney U para Densidades:</strong><br>
            Teste não-paramétrico que compara medianas de densidade entre simulações.<br>
            Não assume distribuição normal dos dados.<br>
            Baseado na comparação de rankings entre as amostras.
            """,
            
            # Métricas Temporais
            "temporal_metrics": """
            <strong>Métricas Temporais:</strong><br>
            Calculadas usando janelas de tempo adaptativas baseadas na granularidade dos dados:<br>
            • <em>HTC:</em> Janelas de max(1, duração/100) ticks (otimizado para granularidade fina)<br>
            • <em>Interscsimulator:</em> Janelas de max(30, duração/50) ticks (otimizado para granularidade maior)<br>
            Para cada janela calcula-se: velocidade média, densidade média, número de veículos únicos.
            """,
            
            # Métricas Básicas
            "basic_metrics": """
            <strong>Métricas Básicas:</strong><br>
            • <em>Total de Veículos:</em> Contagem de IDs únicos de veículos<br>
            • <em>Distância Total:</em> Soma das distâncias percorridas (eventos 'journey_completed')<br>
            • <em>Velocidade Média:</em> Média das velocidades calculadas em eventos 'enter_link'<br>
            • <em>Tempo de Viagem Médio:</em> Diferença temporal entre primeiro e último evento por veículo<br>
            • <em>Throughput:</em> Veículos únicos / duração da simulação
            """,
            
            # Diferenças
            "differences": """
            <strong>Cálculo de Diferenças:</strong><br>
            Para cada métrica M: |M_htc - M_ref| / max(|M_htc|, |M_ref|)<br>
            Normalização permite comparar métricas de diferentes escalas.<br>
            Valores próximos a 0 indicam alta similaridade, próximos a 1 indicam alta diferença.
            """
        }

    def generate_summary_report(self, 
                              comparison_result: ComparisonResult,
                              plots_paths: List[str],
                              save_path: Optional[str] = None) -> str:
        """Gera relatório HTML com resumo da análise"""
        
        descriptions = self._get_metric_descriptions()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório de Comparação de Simulações</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 20px 0; }}
                .methodology {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #2196F3; border-radius: 5px; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .plot img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .collapsible {{ background-color: #777; color: white; cursor: pointer; padding: 10px; width: 100%; border: none; text-align: left; outline: none; font-size: 14px; }}
                .active, .collapsible:hover {{ background-color: #555; }}
                .content {{ padding: 0 15px; display: none; overflow: hidden; background-color: #f9f9f9; }}
                code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-family: monospace; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
            <script>
                function toggleContent(element) {{
                    element.classList.toggle("active");
                    var content = element.nextElementSibling;
                    if (content.style.display === "block") {{
                        content.style.display = "none";
                    }} else {{
                        content.style.display = "block";
                    }}
                }}
            </script>
        </head>
        <body>
            <div class="header">
                <h1>Relatório de Comparação de Simulações</h1>
                <p><strong>Simulação HTC:</strong> {comparison_result.htc_simulation_id}</p>
                <p><strong>Simulação Interscsimulator:</strong> {comparison_result.interscsimulator_simulation_id}</p>
                <p><strong>Data de Geração:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Resumo Executivo</h2>
                <div class="metric">
                    <strong>Score de Similaridade:</strong> {comparison_result.similarity_score:.3f} ({comparison_result.similarity_score*100:.1f}%)
                </div>
                <div class="metric">
                    <strong>Score de Reprodutibilidade:</strong> {comparison_result.reproducibility_score:.3f} ({comparison_result.reproducibility_score*100:.1f}%)
                </div>
                
                <button class="collapsible" onclick="toggleContent(this)">📊 Metodologia dos Scores Principais</button>
                <div class="content">
                    <div class="methodology">
                        {descriptions['similarity_score']}
                    </div>
                    <div class="methodology">
                        {descriptions['reproducibility_score']}
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Métricas de Correlação</h2>
                <table>
                    <tr><th>Métrica</th><th>Valor</th><th>Interpretação</th></tr>
        """
        
        for metric, value in comparison_result.correlation_metrics.items():
            # Determinar interpretação baseada no valor
            if abs(value) >= 0.8:
                interpretation = "Correlação muito forte"
            elif abs(value) >= 0.6:
                interpretation = "Correlação forte"
            elif abs(value) >= 0.4:
                interpretation = "Correlação moderada"
            elif abs(value) >= 0.2:
                interpretation = "Correlação fraca"
            else:
                interpretation = "Correlação muito fraca/inexistente"
                
            if value < 0:
                interpretation += " (negativa)"
            
            html_content += f"<tr><td>{metric}</td><td>{value:.4f}</td><td>{interpretation}</td></tr>"
        
        html_content += """
                </table>
                
                <button class="collapsible" onclick="toggleContent(this)">📈 Metodologia das Correlações</button>
                <div class="content">
                    <div class="methodology">
        """
        
        html_content += descriptions['pearson']
        html_content += """
                    </div>
                    <div class="methodology">
        """
        html_content += descriptions['spearman']
        html_content += """
                    </div>
                    <div class="methodology">
        """
        html_content += descriptions['temporal_metrics']
        html_content += """
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Diferenças Principais</h2>
                <table>
                    <tr><th>Métrica</th><th>Diferença Normalizada</th><th>Avaliação</th></tr>
        """
        
        for diff_metric, value in comparison_result.differences.items():
            if value <= 0.1:
                assessment = "Muito Similar"
            elif value <= 0.3:
                assessment = "Similar"
            elif value <= 0.5:
                assessment = "Moderadamente Diferente"
            elif value <= 0.7:
                assessment = "Diferente"
            else:
                assessment = "Muito Diferente"
            
            html_content += f"<tr><td>{diff_metric}</td><td>{value:.4f}</td><td>{assessment}</td></tr>"
        
        html_content += """
                </table>
                
                <button class="collapsible" onclick="toggleContent(this)">🔍 Metodologia do Cálculo de Diferenças</button>
                <div class="content">
                    <div class="methodology">
        """
        html_content += descriptions['differences']
        html_content += """
                    </div>
                    <div class="methodology">
        """
        html_content += descriptions['basic_metrics']
        html_content += """
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Testes Estatísticos</h2>
                <table>
                    <tr><th>Teste</th><th>Estatística</th><th>P-valor</th><th>Interpretação</th></tr>
        """
        
        for test_name, test_result in comparison_result.statistical_tests.items():
            if isinstance(test_result, dict):
                stat = test_result.get('statistic', 'N/A')
                p_val = test_result.get('p_value', 'N/A')
                
                # Interpretação do p-valor
                if p_val != 'N/A':
                    if p_val < 0.001:
                        interpretation = "Diferença altamente significativa (p < 0.001)"
                    elif p_val < 0.01:
                        interpretation = "Diferença muito significativa (p < 0.01)"
                    elif p_val < 0.05:
                        interpretation = "Diferença significativa (p < 0.05)"
                    elif p_val < 0.1:
                        interpretation = "Diferença marginalmente significativa (p < 0.1)"
                    else:
                        interpretation = "Sem diferença significativa (p ≥ 0.1)"
                else:
                    interpretation = "N/A"
                
                html_content += f"<tr><td>{test_name}</td><td>{stat:.4f}</td><td>{p_val:.4f}</td><td>{interpretation}</td></tr>"
        
        html_content += """
                </table>
                
                <button class="collapsible" onclick="toggleContent(this)">🔬 Metodologia dos Testes Estatísticos</button>
                <div class="content">
                    <div class="methodology">
        """
        html_content += descriptions['speed_t_test']
        html_content += """
                    </div>
                    <div class="methodology">
        """
        html_content += descriptions['speed_ks_test']
        html_content += """
                    </div>
                    <div class="methodology">
        """
        html_content += descriptions['density_mannwhitney']
        html_content += """
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Considerações Metodológicas</h2>
                <div class="warning">
                    <h3>⚠️ Adaptações Específicas para Simuladores</h3>
                    <p><strong>Processamento Temporal Adaptativo:</strong> O sistema detecta automaticamente o tipo de simulador e ajusta as janelas temporais baseado na granularidade dos dados:</p>
                    <ul>
                        <li><strong>HTC:</strong> Usa janelas pequenas (granularidade fina) otimizadas para alta resolução temporal</li>
                        <li><strong>Interscsimulator:</strong> Usa janelas maiores (granularidade maior) para suavização temporal</li>
                    </ul>
                    <p>Esta adaptação resolve problemas de visualização onde dados HTC apareciam como "zig-zag" devido ao uso de janelas temporais inadequadas para sua granularidade específica. Ambos simuladores usam ticks como unidade de tempo, mas com escalas diferentes.</p>
                </div>
                
                <div class="warning">
                    <h3>📝 Limitações e Interpretação</h3>
                    <ul>
                        <li><strong>Normalização de IDs:</strong> Assume mapeamento 1:1 entre veículos das simulações</li>
                        <li><strong>Sincronização Temporal:</strong> Alinhamento baseado em timestamps pode introduzir pequenas discrepâncias</li>
                        <li><strong>Significância Estatística:</strong> P-valores < 0.05 indicam diferenças estatisticamente detectáveis, mas não necessariamente práticas</li>
                        <li><strong>Tamanho da Amostra:</strong> Resultados mais confiáveis com maior número de eventos e veículos</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>Gráficos de Análise</h2>
        """
        
        for plot_path in plots_paths:
            if Path(plot_path).exists():
                plot_name = Path(plot_path).stem.replace('_', ' ').title()
                html_content += f"""
                <div class="plot">
                    <h3>{plot_name}</h3>
                    <img src="{plot_path}" alt="{plot_name}">
                </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        if save_path is None:
            save_path = self.output_dir / 'comparison_report.html'
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Relatório HTML salvo em: {save_path}")
        return str(save_path)


# Exemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    visualizer = SimulationVisualizer()
    print("Sistema de visualização inicializado")