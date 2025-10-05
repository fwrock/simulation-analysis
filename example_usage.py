"""
Exemplo de uso do sistema de análise de simulações
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_extraction.htc_extractor import HTCDataExtractor
from src.data_extraction.interscsimulator_extractor import InterscsimulatorDataExtractor
from src.metrics.calculator import MetricsCalculator
from src.comparison.simulator_comparator import SimulationComparator
from src.visualization.plotter import SimulationVisualizer


def exemplo_analise_htc():
    """Exemplo de análise de simulação HTC"""
    
    print("=== Análise de Simulação HTC ===")
    
    try:
        # Conectar ao Cassandra
        with HTCDataExtractor() as extractor:
            # Listar simulações disponíveis
            sim_ids = extractor.get_simulation_ids()
            print(f"Simulações HTC disponíveis: {sim_ids}")
            
            if not sim_ids:
                print("Nenhuma simulação HTC encontrada")
                return
            
            # Analisar primeira simulação
            sim_id = sim_ids[0]
            print(f"\nAnalisando simulação: {sim_id}")
            
            # Extrair eventos
            events = extractor.get_events_by_simulation(sim_id, limit=1000)
            print(f"Eventos extraídos: {len(events)}")
            
            # Calcular métricas
            calculator = MetricsCalculator()
            basic_metrics = calculator.calculate_basic_metrics(events, sim_id)
            
            print(f"\n--- Métricas Básicas ---")
            print(f"Total de veículos: {basic_metrics.total_vehicles}")
            print(f"Distância total: {basic_metrics.total_distance:.2f} m")
            print(f"Velocidade média: {basic_metrics.average_speed:.2f} m/s")
            print(f"Tempo médio de viagem: {basic_metrics.average_travel_time:.2f} s")
            print(f"Throughput: {basic_metrics.throughput:.4f} veículos/s")
            
            # Métricas de tráfego
            traffic_metrics = calculator.calculate_traffic_metrics(events)
            print(f"\n--- Métricas de Tráfego ---")
            print(f"Densidade média de links: {traffic_metrics.average_link_density:.3f}")
            print(f"Índice de congestionamento: {traffic_metrics.congestion_index:.3f}")
            print(f"Variância da velocidade: {traffic_metrics.speed_variance:.3f}")
            
            # Análise temporal
            temporal_metrics = calculator.calculate_time_series_metrics(events, 300)
            print(f"\n--- Análise Temporal ---")
            print(f"Pontos temporais: {len(temporal_metrics)}")
            if not temporal_metrics.empty:
                print(f"Veículos únicos no primeiro período: {temporal_metrics['unique_vehicles'].iloc[0]}")
                print(f"Velocidade média no primeiro período: {temporal_metrics['avg_speed'].iloc[0]:.2f} m/s")
            
            # Análise por link (primeiros 5)
            link_metrics = calculator.calculate_link_metrics(events)
            print(f"\n--- Análise por Link (primeiros 5) ---")
            for i, link in enumerate(link_metrics[:5]):
                print(f"Link {link.link_id}: densidade={link.average_density:.3f}, "
                     f"velocidade={link.average_speed:.2f} m/s")
            
    except Exception as e:
        print(f"Erro na análise HTC: {e}")


def exemplo_analise_interscsimulator():
    """Exemplo de análise de simulação Interscsimulator"""
    
    print("\n=== Análise de Simulação Interscsimulator ===")
    
    try:
        # Inicializar extrator
        extractor = InterscsimulatorDataExtractor()
        
        # Criar arquivo de exemplo se não houver dados
        sample_file = extractor.data_dir / "example_simulation.xml"
        if not sample_file.exists():
            print("Criando arquivo XML de exemplo...")
            extractor.create_sample_xml(sample_file, 500)
        
        # Listar simulações
        sim_ids = extractor.get_simulation_ids()
        print(f"Simulações Interscsimulator disponíveis: {sim_ids}")
        
        if not sim_ids:
            print("Nenhuma simulação Interscsimulator encontrada")
            return
        
        # Analisar primeira simulação
        sim_id = sim_ids[0]
        print(f"\nAnalisando simulação: {sim_id}")
        
        # Extrair eventos
        events = extractor.get_events_by_simulation(sim_id)
        print(f"Eventos extraídos: {len(events)}")
        
        # Calcular métricas
        calculator = MetricsCalculator()
        basic_metrics = calculator.calculate_basic_metrics(events, sim_id)
        
        print(f"\n--- Métricas Básicas ---")
        print(f"Total de veículos: {basic_metrics.total_vehicles}")
        print(f"Distância total: {basic_metrics.total_distance:.2f} m")
        print(f"Velocidade média: {basic_metrics.average_speed:.2f} m/s")
        print(f"Throughput: {basic_metrics.throughput:.4f} veículos/s")
        
    except Exception as e:
        print(f"Erro na análise Interscsimulator: {e}")


def exemplo_comparacao():
    """Exemplo de comparação entre simuladores"""
    
    print("\n=== Comparação entre Simuladores ===")
    
    try:
        # Dados de exemplo (normalmente viriam dos extractors)
        print("Para uma comparação real, você precisaria de:")
        print("1. Simulação HTC rodando no Cassandra")
        print("2. Arquivo XML do Interscsimulator")
        print("3. IDs de simulação correspondentes")
        
        # Exemplo conceitual de uso
        comparator = SimulationComparator()
        
        print(f"\nNormalizador de IDs configurado:")
        print(f"Padrões HTC carros: {comparator.id_normalizer.htc_car_patterns}")
        print(f"Padrões HTC links: {comparator.id_normalizer.htc_link_patterns}")
        print(f"Padrões referência: {comparator.id_normalizer.ref_car_patterns}")
        
        # Exemplo de normalização
        htc_car_id = "htcaid_car_trip_317"
        normalized = comparator.id_normalizer.normalize_car_id(htc_car_id, 'htc')
        print(f"\nExemplo normalização: {htc_car_id} → {normalized}")
        
        htc_link_id = "htcaid_link_2114"
        normalized_link = comparator.id_normalizer.normalize_link_id(htc_link_id, 'htc')
        print(f"Exemplo normalização: {htc_link_id} → {normalized_link}")
        
    except Exception as e:
        print(f"Erro na comparação: {e}")


def exemplo_visualizacao():
    """Exemplo de geração de visualizações"""
    
    print("\n=== Sistema de Visualização ===")
    
    try:
        # Inicializar visualizador
        visualizer = SimulationVisualizer()
        print(f"Visualizador inicializado. Diretório de saída: {visualizer.output_dir}")
        
        # Exemplo de dados para visualização
        import pandas as pd
        import numpy as np
        
        # Criar dados temporais de exemplo
        n_points = 50
        time_data = pd.DataFrame({
            'time': np.arange(0, n_points * 300, 300),
            'unique_vehicles': np.random.poisson(20, n_points),
            'avg_speed': np.random.normal(12, 2, n_points),
            'avg_density': np.random.uniform(0.1, 0.8, n_points),
            'total_events': np.random.poisson(100, n_points)
        })
        
        # Gerar gráfico temporal
        plot_path = visualizer.plot_temporal_metrics(time_data, "Exemplo - Métricas Temporais")
        print(f"Gráfico temporal salvo: {plot_path}")
        
        # Criar dados de densidade para mapa de calor
        density_data = pd.DataFrame({
            'timestamp': np.tile(np.arange(0, 3000, 300), 5),
            'link_id': np.repeat(['link_1', 'link_2', 'link_3', 'link_4', 'link_5'], 10),
            'density': np.random.uniform(0, 1, 50)
        })
        
        # Gerar mapa de calor
        heatmap_path = visualizer.plot_density_heatmap(density_data, "Exemplo - Densidade de Links")
        print(f"Mapa de calor salvo: {heatmap_path}")
        
        print(f"\nArquivos gerados no diretório: {visualizer.output_dir}")
        
    except Exception as e:
        print(f"Erro na visualização: {e}")


def exemplo_metricas_especificas():
    """Exemplo de cálculo de métricas específicas"""
    
    print("\n=== Métricas Específicas ===")
    
    try:
        # Conectar ao HTC para exemplo real
        with HTCDataExtractor() as extractor:
            sim_ids = extractor.get_simulation_ids()
            
            if not sim_ids:
                print("Nenhuma simulação HTC para análise específica")
                return
            
            sim_id = sim_ids[0]
            print(f"Analisando métricas específicas para: {sim_id}")
            
            # Buscar eventos de um veículo específico
            all_events = extractor.get_events_by_simulation(sim_id, limit=500)
            
            if not all_events:
                print("Nenhum evento encontrado")
                return
            
            # Encontrar primeiro veículo
            first_car = None
            for event in all_events:
                if event.car_id:
                    first_car = event.car_id
                    break
            
            if first_car:
                print(f"\nAnalisando veículo: {first_car}")
                
                # Extrair jornada do veículo
                vehicle_events = extractor.get_vehicle_journey(sim_id, first_car)
                print(f"Eventos do veículo: {len(vehicle_events)}")
                
                # Calcular métricas do veículo
                calculator = MetricsCalculator()
                vehicle_metrics = calculator.calculate_vehicle_metrics(vehicle_events, first_car)
                
                print(f"--- Métricas do Veículo {first_car} ---")
                for metric, value in vehicle_metrics.items():
                    print(f"{metric}: {value}")
            
            # Análise de densidade de links
            print(f"\n--- Análise de Densidade de Links ---")
            density_data = extractor.get_link_density_data(sim_id)
            
            if not density_data.empty:
                print(f"Registros de densidade: {len(density_data)}")
                print(f"Links únicos: {density_data['link_id'].nunique()}")
                print(f"Densidade média geral: {density_data['density'].mean():.3f}")
                print(f"Densidade máxima: {density_data['density'].max():.3f}")
                
                # Top 5 links mais congestionados
                top_congested = density_data.groupby('link_id')['density'].mean().sort_values(ascending=False).head(5)
                print(f"\nTop 5 links mais congestionados:")
                for link_id, avg_density in top_congested.items():
                    print(f"  {link_id}: {avg_density:.3f}")
            
    except Exception as e:
        print(f"Erro nas métricas específicas: {e}")


def main():
    """Executa todos os exemplos"""
    
    print("Sistema de Análise de Simulações de Tráfego Urbano")
    print("=" * 60)
    
    # Executar exemplos
    exemplo_analise_htc()
    exemplo_analise_interscsimulator()
    exemplo_comparacao()
    exemplo_visualizacao()
    exemplo_metricas_especificas()
    
    print("\n" + "=" * 60)
    print("Exemplos concluídos!")
    print("\nPara usar o sistema completo:")
    print("python main.py --mode compare --htc-sim SUA_SIM_HTC --ref-sim SUA_SIM_REF")


if __name__ == "__main__":
    main()