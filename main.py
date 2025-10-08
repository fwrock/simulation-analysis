"""
Script principal para análise de simulações de tráfego urbano
Compara simulações entre HTC e Interscsimulator
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import json

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent))

from src.data_extraction.htc_extractor import HTCDataExtractor
from src.data_extraction.htc_jsonl_extractor import HTCJsonlExtractor
from src.data_extraction.interscsimulator_extractor import InterscsimulatorDataExtractor
from src.metrics.calculator import MetricsCalculator
from src.comparison.simulator_comparator import SimulationComparator
from src.visualization.plotter import SimulationVisualizer

# Tentar importar configurações
try:
    from config.settings import OUTPUT_CONFIG
except ImportError:
    OUTPUT_CONFIG = {'base_dir': './output'}


def setup_logging(log_level: str = 'INFO'):
    """Configura sistema de logging"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation_analysis.log'),
            logging.StreamHandler()
        ]
    )


class SimulationAnalyzer:
    """Analisador principal de simulações"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(OUTPUT_CONFIG['base_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar componentes
        self.htc_extractor = HTCDataExtractor()
        self.interscs_extractor = InterscsimulatorDataExtractor()
        self.metrics_calculator = MetricsCalculator()
        self.comparator = SimulationComparator()
        self.visualizer = SimulationVisualizer(self.output_dir / 'plots')
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_single_simulation(self, simulation_id: str, simulator_type: str):
        """Analisa uma simulação específica"""
        
        self.logger.info(f"Analisando simulação {simulation_id} do {simulator_type}")
        
        # Extrair dados
        if simulator_type.lower() == 'htc':
            if not self.htc_extractor.connect():
                raise ConnectionError("Não foi possível conectar ao Cassandra")
            
            events = self.htc_extractor.get_events_by_simulation(simulation_id)
            summary = self.htc_extractor.get_simulation_summary(simulation_id)
            
            self.htc_extractor.disconnect()
        
        elif simulator_type.lower() == 'interscsimulator':
            # Para Interscsimulator, simulation_id é o caminho do arquivo
            events = self.interscs_extractor.get_events_by_simulation(simulation_id)
            summary = self.interscs_extractor.get_simulation_summary(simulation_id)
            # Usar o nome do arquivo como ID para relatórios
            simulation_id = Path(simulation_id).stem
        
        else:
            raise ValueError(f"Simulador não suportado: {simulator_type}")
        
        if not events:
            self.logger.warning(f"Nenhum evento encontrado para simulação {simulation_id}")
            return
        
        # Calcular métricas
        basic_metrics = self.metrics_calculator.calculate_basic_metrics(events, simulation_id)
        traffic_metrics = self.metrics_calculator.calculate_traffic_metrics(events)
        link_metrics = self.metrics_calculator.calculate_link_metrics(events)
        temporal_metrics = self.metrics_calculator.calculate_time_series_metrics(events)
        
        # Gerar visualizações
        plots = []
        
        # Gráfico temporal
        if not temporal_metrics.empty:
            temporal_plot = self.visualizer.plot_temporal_metrics(
                temporal_metrics, 
                f"Métricas Temporais - {simulation_id}"
            )
            plots.append(temporal_plot)
        
        # Mapa de calor de densidade
        if simulator_type.lower() == 'htc':
            density_data = self.htc_extractor.get_link_density_data(simulation_id)
        else:
            # Para Interscsimulator, usar o caminho original do arquivo
            original_path = summary.get('file_path', simulation_id)
            density_data = self.interscs_extractor.get_link_density_data(original_path)
        
        if not density_data.empty:
            heatmap_plot = self.visualizer.plot_density_heatmap(
                density_data,
                f"Densidade de Links - {simulation_id}"
            )
            plots.append(heatmap_plot)
        
        # Salvar resultados
        results = {
            'simulation_id': simulation_id,
            'simulator_type': simulator_type,
            'summary': summary,
            'basic_metrics': basic_metrics.__dict__,
            'traffic_metrics': traffic_metrics.__dict__,
            'link_metrics_count': len(link_metrics),
            'temporal_metrics_points': len(temporal_metrics),
            'generated_plots': plots
        }
        
        results_file = self.output_dir / f'analysis_{simulation_id}_{simulator_type}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Análise da simulação {simulation_id} concluída")
        self.logger.info(f"Resultados salvos em: {results_file}")
        
        return results
    
    def compare_simulations(self, htc_sim_id: str, ref_sim_path: str):
        """Compara duas simulações"""
        
        self.logger.info(f"🚀 Comparando simulações: HTC {htc_sim_id} vs Interscsimulator {ref_sim_path}")
        
        # Extrair dados do HTC
        self.logger.info("🔗 Conectando ao Cassandra...")
        if not self.htc_extractor.connect():
            raise ConnectionError("Não foi possível conectar ao Cassandra")
        
        self.logger.info(f"📊 Extraindo eventos HTC da simulação {htc_sim_id}...")
        htc_events = self.htc_extractor.get_events_by_simulation(htc_sim_id)
        
        self.logger.info(f"📋 Extraindo resumo HTC da simulação {htc_sim_id}...")
        htc_summary = self.htc_extractor.get_simulation_summary(htc_sim_id)
        
        self.logger.info("🔚 Desconectando do Cassandra...")
        self.htc_extractor.disconnect()
        
        # Extrair dados do Interscsimulator
        self.logger.info(f"📊 Extraindo eventos Interscsimulator de {ref_sim_path}...")
        ref_events = self.interscs_extractor.get_events_by_simulation(ref_sim_path)
        
        self.logger.info(f"📋 Extraindo resumo Interscsimulator de {ref_sim_path}...")
        ref_summary = self.interscs_extractor.get_simulation_summary(ref_sim_path)
        ref_sim_id = Path(ref_sim_path).stem  # Usar nome do arquivo como ID
        
        if not htc_events or not ref_events:
            raise ValueError("Uma ou ambas simulações não possuem eventos")
        
        self.logger.info(f"✅ Dados extraídos:")
        self.logger.info(f"   📊 HTC: {len(htc_events):,} eventos")
        self.logger.info(f"   📊 Interscsimulator: {len(ref_events):,} eventos")
        
        # Realizar comparação
        self.logger.info("🔄 Iniciando comparação detalhada...")
        comparison_result = self.comparator.compare_simulations(
            htc_events, ref_events, htc_sim_id, ref_sim_id
        )
        
        # Calcular métricas para visualização
        self.logger.info("📈 Calculando métricas para visualização...")
        htc_metrics = self.metrics_calculator.calculate_basic_metrics(htc_events, htc_sim_id)
        ref_metrics = self.metrics_calculator.calculate_basic_metrics(ref_events, ref_sim_id)
        
        self.logger.info("📊 Calculando métricas temporais...")
        htc_temporal = self.metrics_calculator.calculate_time_series_metrics(htc_events)
        ref_temporal = self.metrics_calculator.calculate_time_series_metrics(ref_events)
        
        # Gerar visualizações
        self.logger.info("🎨 Iniciando geração de visualizações...")
        plots = []
        
        # Criar análise completa com todos os novos gráficos
        comprehensive_plots = self.visualizer.create_comprehensive_analysis(
            htc_events, ref_events, top_n_links=20
        )
        plots.extend(comprehensive_plots.values())
        
        # Dashboard interativo
        self.logger.info("📊 Criando dashboard interativo...")
        dashboard_path = self.visualizer.create_interactive_dashboard(
            comparison_result, htc_temporal, ref_temporal
        )
        plots.append(dashboard_path)
        
        # Relatório final
        self.logger.info("📄 Gerando relatório final...")
        report_path = self.visualizer.generate_summary_report(
            comparison_result, plots
        )
        
        # Salvar resultados da comparação
        self.logger.info("💾 Salvando resultados...")
        comparison_data = {
            'htc_simulation': htc_sim_id,
            'ref_simulation': ref_sim_id,
            'ref_file_path': str(ref_sim_path),
            'comparison_result': {
                'similarity_score': comparison_result.similarity_score,
                'reproducibility_score': comparison_result.reproducibility_score,
                'statistical_tests': comparison_result.statistical_tests,
                'correlation_metrics': comparison_result.correlation_metrics,
                'differences': comparison_result.differences
            },
            'htc_summary': htc_summary,
            'ref_summary': ref_summary,
            'generated_plots': plots,
            'report_path': report_path
        }
        
        results_file = self.output_dir / f'comparison_{htc_sim_id}_vs_{ref_sim_id}.json'
        with open(results_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        self.logger.info(f"Comparação concluída")
        self.logger.info(f"Resultados salvos em: {results_file}")
        self.logger.info(f"Relatório HTML: {report_path}")
        
        return comparison_data
    
    def compare_with_jsonl(self, htc_jsonl_path: str, ref_sim_path: str):
        """Compara simulação HTC via JSONL com simulação de referência"""
        
        self.logger.info(f"🚀 Comparando HTC (JSONL) vs Interscsimulator (XML)...")
        self.logger.info(f"📁 HTC JSONL: {htc_jsonl_path}")
        self.logger.info(f"📁 Interscsimulator XML: {ref_sim_path}")
        
        # Verificar se arquivos existem
        jsonl_path = Path(htc_jsonl_path)
        xml_path = Path(ref_sim_path)
        
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Arquivo JSONL não encontrado: {htc_jsonl_path}")
        if not xml_path.exists():
            raise FileNotFoundError(f"Arquivo XML não encontrado: {ref_sim_path}")
        
        # Extrair dados do Interscsimulator
        self.logger.info("📊 Extraindo dados do Interscsimulator...")
        ref_events = self.interscs_extractor.get_events_by_simulation(ref_sim_path)
        ref_summary = self.interscs_extractor.get_simulation_summary(ref_sim_path)
        
        # Executar comparação com JSONL
        self.logger.info("🔄 Executando comparação com dados JSONL...")
        ref_sim_id = Path(ref_sim_path).stem
        comparison_result = self.comparator.compare_with_jsonl_data(
            htc_jsonl_path, ref_events, ref_sim_id
        )
        
        # Gerar visualizações
        self.logger.info("📊 Gerando visualizações...")
        plots = []
        
        # Extrair dados do JSONL para visualização
        jsonl_extractor = HTCJsonlExtractor(htc_jsonl_path)
        htc_df, htc_stats = jsonl_extractor.extract_events()
        
        # Converter para eventos para compatibilidade com visualizador
        htc_events = self.comparator._dataframe_to_events(htc_df, 'htc')
        
        # Análise abrangente
        self.logger.info("📈 Criando análise abrangente...")
        comprehensive_plots = self.visualizer.create_comprehensive_analysis(
            htc_events, ref_events
        )
        plots.extend(comprehensive_plots.values())
        
        # Dashboard interativo
        self.logger.info("📊 Criando dashboard interativo...")
        # Calcular métricas temporais para dashboard
        htc_temporal = self.metrics_calculator.calculate_time_series_metrics(htc_events)
        ref_temporal = self.metrics_calculator.calculate_time_series_metrics(ref_events)
        
        dashboard_path = self.visualizer.create_interactive_dashboard(
            comparison_result['standard_comparison'], htc_temporal, ref_temporal
        )
        plots.append(dashboard_path)
        
        # Relatório final
        self.logger.info("📄 Gerando relatório final...")
        report_path = self.visualizer.generate_summary_report(
            comparison_result['standard_comparison'], plots
        )
        
        # Relatório específico para análise JSONL
        self.logger.info("📋 Gerando relatório específico de análise JSONL...")
        jsonl_report_path = self._generate_jsonl_analysis_report(comparison_result, plots)
        
        # Salvar resultados
        self.logger.info("💾 Salvando resultados...")
        jsonl_sim_id = htc_stats.get('simulation_id', jsonl_path.stem)
        comparison_data = {
            'htc_jsonl_path': str(htc_jsonl_path),
            'ref_simulation': ref_sim_id,
            'ref_file_path': str(ref_sim_path),
            'standard_comparison': {
                'similarity_score': comparison_result['standard_comparison'].similarity_score,
                'reproducibility_score': comparison_result['standard_comparison'].reproducibility_score,
                'statistical_tests': comparison_result['standard_comparison'].statistical_tests,
                'correlation_metrics': comparison_result['standard_comparison'].correlation_metrics,
                'differences': comparison_result['standard_comparison'].differences
            },
            'routes_analysis': comparison_result.get('routes_analysis', {}),
            'events_analysis': comparison_result.get('events_analysis', {}),
            'temporal_analysis': comparison_result.get('temporal_analysis', {}),
            'htc_stats': comparison_result.get('htc_stats', {}),
            'ref_summary': ref_summary,
            'generated_plots': plots,
            'standard_report_path': report_path,
            'jsonl_report_path': jsonl_report_path
        }
        
        results_file = self.output_dir / f'comparison_jsonl_{jsonl_sim_id}_vs_{ref_sim_id}.json'
        with open(results_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        self.logger.info("✅ Comparação com JSONL concluída!")
        self.logger.info(f"📄 Resultados salvos em: {results_file}")
        self.logger.info(f"📊 Relatório padrão: {report_path}")
        self.logger.info(f"📋 Relatório JSONL: {jsonl_report_path}")
        
        return comparison_data
    
    def _generate_jsonl_analysis_report(self, comparison_result: dict, plots: list) -> str:
        """Gera relatório específico para análise JSONL"""
        
        report_path = self.output_dir / 'jsonl_analysis_report.html'
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório de Análise JSONL - HTC vs Interscsimulator</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #2196F3; background-color: #f9f9f9; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #2196F3; color: white; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .plot img {{ max-width: 100%; height: auto; }}
                .success {{ color: #4CAF50; font-weight: bold; }}
                .warning {{ color: #FF9800; font-weight: bold; }}
                .error {{ color: #F44336; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Relatório de Análise JSONL</h1>
                <h2>HTC (JSONL) vs Interscsimulator (XML)</h2>
                <p><strong>Data de Geração:</strong> {Path().resolve()}</p>
            </div>
            
            <div class="section">
                <h2>🗺️ Análise de Rotas</h2>
                
                <button class="collapsible" onclick="toggleContent(this)">📚 Metodologia da Análise de Rotas</button>
                <div class="content">
                    <div class="methodology">
                        <h3>🔬 Como São Calculadas as Métricas de Rotas</h3>
                        <p><strong>Processo de Comparação:</strong></p>
                        <ol>
                            <li><strong>Pareamento de Veículos:</strong> Identifica veículos correspondentes entre os dois simuladores através de normalização de IDs</li>
                            <li><strong>Extração de Rotas:</strong> Para cada veículo, extrai informações de rota (comprimento, custo, links utilizados)</li>
                            <li><strong>Cálculo de Diferenças:</strong> Para cada par de rotas correspondentes:<br>
                                <code>Diferença = |Valor_HTC - Valor_Interscsimulator| / max(Valor_HTC, Valor_Interscsimulator)</code></li>
                        </ol>
                        
                        <p><strong>Significado das Métricas:</strong></p>
                        <ul>
                            <li><strong>Comprimento da Rota:</strong> Distância total percorrida (metros/km)</li>
                            <li><strong>Custo da Rota:</strong> Tempo total estimado ou custo computacional</li>
                            <li><strong>Complexidade da Rota:</strong> Número de links (segmentos) na rota</li>
                        </ul>
                        
                        <p><strong>Interpretação dos Valores:</strong></p>
                        <ul>
                            <li><strong>Valor Médio:</strong> Média aritmética das diferenças entre <em>todas</em> as rotas comparáveis - indica tendência geral</li>
                            <li><strong>Valor Máximo:</strong> Maior diferença encontrada entre <em>qualquer</em> par de rotas - indica pior caso</li>
                            <li><strong>Diferença Médio vs Máximo:</strong> Grande diferença sugere presença de outliers (algumas rotas muito discrepantes)</li>
                        </ul>
                        
                        <p><strong>Critérios de Avaliação:</strong></p>
                        <ul>
                            <li>🟢 <strong>Excelente (< 5%):</strong> Diferenças mínimas, rotas muito similares</li>
                            <li>🟡 <strong>Aceitável (5-15%):</strong> Diferenças moderadas, rotas razoavelmente similares</li>
                            <li>🔴 <strong>Significativa (> 15%):</strong> Diferenças importantes, rotas substancialmente diferentes</li>
                        </ul>
                    </div>
                </div>
        """
        
        if 'routes_analysis' in comparison_result:
            routes = comparison_result['routes_analysis']
            html_content += f"""
                <div class="metric">
                    <strong>📊 Rotas HTC:</strong> {routes.get('htc_routes_count', 0):,}<br>
                    <strong>📊 Rotas Interscsimulator:</strong> {routes.get('ref_routes_count', 0):,}<br>
                    <strong>🔗 Veículos Correspondentes:</strong> {routes.get('matching_vehicles', 0):,}
                </div>
                
                <table>
                    <tr><th>Métrica</th><th>Valor Médio</th><th>Valor Máximo</th><th>Avaliação</th></tr>
            """
            
            # Diferenças de comprimento
            if 'avg_length_difference' in routes:
                avg_len = routes['avg_length_difference']
                max_len = routes.get('max_length_difference', 0)
                assessment = "🟢 Excelente" if avg_len < 0.05 else "🟡 Aceitável" if avg_len < 0.15 else "🔴 Significativa"
                html_content += f"<tr><td>Diferença de Comprimento</td><td>{avg_len:.2%}</td><td>{max_len:.2%}</td><td>{assessment}</td></tr>"
            
            # Diferenças de custo
            if 'avg_cost_difference' in routes:
                avg_cost = routes['avg_cost_difference']
                max_cost = routes.get('max_cost_difference', 0)
                assessment = "🟢 Excelente" if avg_cost < 0.05 else "🟡 Aceitável" if avg_cost < 0.15 else "🔴 Significativa"
                html_content += f"<tr><td>Diferença de Custo</td><td>{avg_cost:.2%}</td><td>{max_cost:.2%}</td><td>{assessment}</td></tr>"
            
            # Diferenças de complexidade
            if 'avg_complexity_difference' in routes:
                avg_comp = routes['avg_complexity_difference']
                max_comp = routes.get('max_complexity_difference', 0)
                assessment = "🟢 Excelente" if avg_comp < 0.1 else "🟡 Aceitável" if avg_comp < 0.25 else "🔴 Significativa"
                html_content += f"<tr><td>Diferença de Complexidade</td><td>{avg_comp:.2%}</td><td>{max_comp:.2%}</td><td>{assessment}</td></tr>"
            
            html_content += "</table>"
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>📊 Análise de Distribuição de Eventos</h2>
        """
        
        if 'events_analysis' in comparison_result:
            events = comparison_result['events_analysis']
            html_content += f"""
                <div class="metric">
                    <strong>📈 Total HTC:</strong> {events.get('htc_total', 0):,}<br>
                    <strong>📈 Total Interscsimulator:</strong> {events.get('ref_total', 0):,}<br>
                    <strong>📊 Diferença:</strong> {events.get('ref_total', 0) - events.get('htc_total', 0):,}
                </div>
                
                <h3>Tipos de Eventos por Simulador</h3>
                <table>
                    <tr><th>Tipo de Evento</th><th>HTC</th><th>Interscsimulator</th><th>Diferença</th><th>% HTC</th><th>% Interscs</th></tr>
            """
            
            event_comparison = events.get('event_type_comparison', {})
            for event_type, data in event_comparison.items():
                htc_count = data.get('htc_count', 0)
                ref_count = data.get('ref_count', 0)
                difference = data.get('difference', 0)
                htc_pct = data.get('htc_percentage', 0)
                ref_pct = data.get('ref_percentage', 0)
                
                html_content += f"""
                    <tr>
                        <td>{event_type}</td>
                        <td>{htc_count:,}</td>
                        <td>{ref_count:,}</td>
                        <td>{difference:,}</td>
                        <td>{htc_pct:.1f}%</td>
                        <td>{ref_pct:.1f}%</td>
                    </tr>
                """
            
            html_content += "</table>"
            
            # Eventos ausentes
            missing_htc = events.get('missing_in_htc', [])
            missing_ref = events.get('missing_in_ref', [])
            
            if missing_htc:
                html_content += f"<p><strong class='warning'>⚠️ Eventos ausentes no HTC:</strong> {', '.join(missing_htc)}</p>"
            
            if missing_ref:
                html_content += f"<p><strong class='warning'>⚠️ Eventos ausentes no Interscsimulator:</strong> {', '.join(missing_ref)}</p>"
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>⏱️ Análise Temporal</h2>
        """
        
        if 'temporal_analysis' in comparison_result:
            temporal = comparison_result['temporal_analysis']
            html_content += f"""
                <table>
                    <tr><th>Métrica Temporal</th><th>HTC</th><th>Interscsimulator</th><th>Diferença</th></tr>
                    <tr><td>Duração (ticks)</td><td>{temporal.get('htc_duration', 0):,}</td><td>{temporal.get('ref_duration', 0):,}</td><td>{temporal.get('duration_difference', 0):,}</td></tr>
                    <tr><td>Eventos por Tick</td><td>{temporal.get('htc_events_per_tick', 0):.2f}</td><td>{temporal.get('ref_events_per_tick', 0):.2f}</td><td>{temporal.get('events_rate_difference', 0):.2f}</td></tr>
                    <tr><td>Pico de Atividade</td><td>{temporal.get('htc_peak_activity', 0):,}</td><td>{temporal.get('ref_peak_activity', 0):,}</td><td>-</td></tr>
                    <tr><td>Atividade Média</td><td>{temporal.get('htc_avg_activity', 0):.1f}</td><td>{temporal.get('ref_avg_activity', 0):.1f}</td><td>-</td></tr>
                </table>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>📊 Gráficos de Análise</h2>
        """
        
        for plot_path in plots:
            if Path(plot_path).exists():
                plot_name = Path(plot_path).stem.replace('_', ' ').title()
                plot_filename = Path(plot_path).name
                html_content += f"""
                <div class="plot">
                    <h3>{plot_name}</h3>
                    <img src="{plot_filename}" alt="{plot_name}">
                </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def compare_vehicle_journey(self, htc_sim_id: str, ref_sim_path: str, vehicle_id: str):
        """Compara jornada de um veículo específico"""
        
        self.logger.info(f"Comparando jornada do veículo {vehicle_id}")
        
        # Extrair dados
        if not self.htc_extractor.connect():
            raise ConnectionError("Não foi possível conectar ao Cassandra")
        
        htc_events = self.htc_extractor.get_events_by_simulation(htc_sim_id)
        self.htc_extractor.disconnect()
        
        ref_events = self.interscs_extractor.get_events_by_simulation(ref_sim_path)
        
        # Comparar jornada
        journey_comparison = self.comparator.compare_vehicle_journeys(
            htc_events, ref_events, vehicle_id
        )
        
        if 'error' in journey_comparison:
            self.logger.error(journey_comparison['error'])
            return journey_comparison
        
        # Extrair eventos específicos do veículo
        normalized_id = self.comparator.id_normalizer.normalize_car_id(vehicle_id, 'htc')
        
        htc_vehicle_events = [e for e in htc_events if 
                             self.comparator.id_normalizer.normalize_car_id(e.car_id, 'htc') == normalized_id]
        ref_vehicle_events = [e for e in ref_events if 
                             self.comparator.id_normalizer.normalize_car_id(e.car_id, 'interscsimulator') == normalized_id]
        
        # Gerar visualização da jornada
        if htc_vehicle_events and ref_vehicle_events:
            journey_plot = self.visualizer.plot_vehicle_journey(
                htc_vehicle_events, ref_vehicle_events, normalized_id
            )
            journey_comparison['plot_path'] = journey_plot
        
        # Salvar resultados
        results_file = self.output_dir / f'vehicle_journey_{normalized_id}.json'
        with open(results_file, 'w') as f:
            json.dump(journey_comparison, f, indent=2, default=str)
        
        self.logger.info(f"Análise de jornada concluída: {results_file}")
        
        return journey_comparison
    
    def batch_analysis(self, htc_sim_ids: List[str], ref_sim_paths: List[str]):
        """Análise em lote de múltiplas simulações"""
        
        self.logger.info(f"Iniciando análise em lote: {len(htc_sim_ids)} HTC, {len(ref_sim_paths)} Interscsimulator")
        
        results = {
            'htc_analyses': [],
            'ref_analyses': [],
            'comparisons': []
        }
        
        # Analisar simulações HTC
        for sim_id in htc_sim_ids:
            try:
                analysis = self.analyze_single_simulation(sim_id, 'htc')
                results['htc_analyses'].append(analysis)
            except Exception as e:
                self.logger.error(f"Erro ao analisar simulação HTC {sim_id}: {e}")
        
        # Analisar simulações Interscsimulator
        for sim_path in ref_sim_paths:
            try:
                analysis = self.analyze_single_simulation(sim_path, 'interscsimulator')
                results['ref_analyses'].append(analysis)
            except Exception as e:
                self.logger.error(f"Erro ao analisar simulação Interscsimulator {sim_path}: {e}")
        
        # Realizar comparações cruzadas
        for htc_id in htc_sim_ids:
            for ref_path in ref_sim_paths:
                try:
                    comparison = self.compare_simulations(htc_id, ref_path)
                    results['comparisons'].append(comparison)
                except Exception as e:
                    ref_id = Path(ref_path).stem
                    self.logger.error(f"Erro ao comparar {htc_id} vs {ref_id}: {e}")
        
        # Salvar resultados consolidados
        batch_results_file = self.output_dir / 'batch_analysis_results.json'
        with open(batch_results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Análise em lote concluída: {batch_results_file}")
        
        return results


def main():
    """Função principal"""
    
    parser = argparse.ArgumentParser(description='Análise de Simulações de Tráfego Urbano')
    parser.add_argument('--mode', choices=['single', 'compare', 'jsonl', 'vehicle', 'batch'], 
                       required=True, help='Modo de análise')
    parser.add_argument('--htc-sim', help='ID da simulação HTC')
    parser.add_argument('--htc-jsonl', help='Caminho do arquivo JSONL do HTC')
    parser.add_argument('--ref-sim', help='Caminho do arquivo XML da simulação Interscsimulator')
    parser.add_argument('--vehicle-id', help='ID do veículo para análise específica')
    parser.add_argument('--simulator', choices=['htc', 'interscsimulator'], 
                       help='Tipo de simulador (para análise single)')
    parser.add_argument('--output-dir', help='Diretório de saída')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--htc-sims', nargs='+', help='Lista de IDs HTC (para batch)')
    parser.add_argument('--ref-sims', nargs='+', help='Lista de caminhos de arquivos XML Interscsimulator (para batch)')
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Inicializar analisador
        analyzer = SimulationAnalyzer(args.output_dir)
        
        if args.mode == 'single':
            if not args.htc_sim and not args.ref_sim:
                raise ValueError("É necessário especificar --htc-sim ou --ref-sim para análise single")
            
            sim_id = args.htc_sim or args.ref_sim
            simulator = args.simulator or ('htc' if args.htc_sim else 'interscsimulator')
            
            analyzer.analyze_single_simulation(sim_id, simulator)
        
        elif args.mode == 'compare':
            if not args.htc_sim or not args.ref_sim:
                raise ValueError("É necessário especificar --htc-sim e --ref-sim para comparação")
            
            analyzer.compare_simulations(args.htc_sim, args.ref_sim)
        
        elif args.mode == 'jsonl':
            if not args.htc_jsonl or not args.ref_sim:
                raise ValueError("É necessário especificar --htc-jsonl e --ref-sim para análise JSONL")
            
            analyzer.compare_with_jsonl(args.htc_jsonl, args.ref_sim)
        
        elif args.mode == 'vehicle':
            if not args.htc_sim or not args.ref_sim or not args.vehicle_id:
                raise ValueError("É necessário especificar --htc-sim, --ref-sim e --vehicle-id")
            
            analyzer.compare_vehicle_journey(args.htc_sim, args.ref_sim, args.vehicle_id)
        
        elif args.mode == 'batch':
            if not args.htc_sims or not args.ref_sims:
                raise ValueError("É necessário especificar --htc-sims e --ref-sims para análise batch")
            
            analyzer.batch_analysis(args.htc_sims, args.ref_sims)
        
        logger.info("Análise concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante a análise: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()