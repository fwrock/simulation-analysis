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
        
        self.logger.info(f"Comparando simulações: HTC {htc_sim_id} vs Interscsimulator {ref_sim_path}")
        
        # Extrair dados do HTC
        if not self.htc_extractor.connect():
            raise ConnectionError("Não foi possível conectar ao Cassandra")
        
        htc_events = self.htc_extractor.get_events_by_simulation(htc_sim_id)
        htc_summary = self.htc_extractor.get_simulation_summary(htc_sim_id)
        self.htc_extractor.disconnect()
        
        # Extrair dados do Interscsimulator
        ref_events = self.interscs_extractor.get_events_by_simulation(ref_sim_path)
        ref_summary = self.interscs_extractor.get_simulation_summary(ref_sim_path)
        ref_sim_id = Path(ref_sim_path).stem  # Usar nome do arquivo como ID
        
        if not htc_events or not ref_events:
            raise ValueError("Uma ou ambas simulações não possuem eventos")
        
        # Realizar comparação
        comparison_result = self.comparator.compare_simulations(
            htc_events, ref_events, htc_sim_id, ref_sim_id
        )
        
        # Calcular métricas para visualização
        htc_metrics = self.metrics_calculator.calculate_basic_metrics(htc_events, htc_sim_id)
        ref_metrics = self.metrics_calculator.calculate_basic_metrics(ref_events, ref_sim_id)
        
        htc_temporal = self.metrics_calculator.calculate_time_series_metrics(htc_events)
        ref_temporal = self.metrics_calculator.calculate_time_series_metrics(ref_events)
        
        # Gerar visualizações
        plots = []
        
        # Comparação de métricas básicas
        basic_comparison_plot = self.visualizer.plot_basic_metrics_comparison(
            htc_metrics, ref_metrics
        )
        plots.append(basic_comparison_plot)
        
        # Distribuição de velocidades
        speed_dist_plot = self.visualizer.plot_speed_distribution(htc_events, ref_events)
        plots.append(speed_dist_plot)
        
        # Comparação de links
        link_comparison_data = self.comparator.calculate_link_density_comparison(htc_events, ref_events)
        if not link_comparison_data.empty:
            link_plot = self.visualizer.plot_link_comparison(link_comparison_data)
            plots.append(link_plot)
        
        # Dashboard interativo
        dashboard_path = self.visualizer.create_interactive_dashboard(
            comparison_result, htc_temporal, ref_temporal
        )
        plots.append(dashboard_path)
        
        # Relatório final
        report_path = self.visualizer.generate_summary_report(
            comparison_result, plots
        )
        
        # Salvar resultados da comparação
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
    parser.add_argument('--mode', choices=['single', 'compare', 'vehicle', 'batch'], 
                       required=True, help='Modo de análise')
    parser.add_argument('--htc-sim', help='ID da simulação HTC')
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