#!/usr/bin/env python3
"""
Script para testar a análise de dados JSONL do HTC
"""

import logging
import sys
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.data_extraction.htc_jsonl_extractor import HTCJsonlExtractor
from src.data_extraction.interscsimulator_extractor import InterscsimulatorDataExtractor
from src.comparison.simulator_comparator import SimulationComparator
from src.visualization.plotter import SimulationVisualizer

def test_jsonl_analysis():
    """Testa a análise de dados JSONL"""
    
    print("🧪 Testando análise de dados JSONL...")
    
    # Exemplo de paths (ajustar conforme necessário)
    htc_jsonl_path = "/home/dean/hyperbolic-time-chamber/output/reports/json/bfs_cenario_2500_viagens_3/events.jsonl"
    ref_xml_path = "data/interscsimulator/events_bfs_cenario_2500_viagens_3.xml"
    
    # Verificar se arquivos existem
    if not Path(htc_jsonl_path).exists():
        print(f"❌ Arquivo JSONL não encontrado: {htc_jsonl_path}")
        print("📝 Usando dados simulados para teste...")
        return test_with_simulated_data()
    
    if not Path(ref_xml_path).exists():
        print(f"❌ Arquivo XML não encontrado: {ref_xml_path}")
        print("📝 Usando dados simulados para teste...")
        return test_with_simulated_data()
    
    try:
        # Extrair dados do JSONL
        print("🔄 Extraindo dados do JSONL...")
        jsonl_extractor = HTCJsonlExtractor(htc_jsonl_path)
        htc_df, htc_stats = jsonl_extractor.extract_events()
        
        # Extrair dados do XML
        print("🔄 Extraindo dados do XML...")
        xml_extractor = InterscsimulatorDataExtractor(ref_xml_path)
        ref_events = xml_extractor.extract_events()
        
        # Comparar
        print("🔄 Executando comparação...")
        comparator = SimulationComparator()
        comparison_result = comparator.compare_with_jsonl_data(
            htc_jsonl_path, ref_events, "ref_simulation"
        )
        
        # Mostrar resultados
        print_comparison_summary(comparison_result)
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_simulated_data():
    """Testa com dados simulados"""
    
    print("🎭 Testando extrator JSONL com dados simulados...")
    
    # Criar arquivo JSONL de teste
    test_jsonl_path = "/tmp/test_events.jsonl"
    
    test_data = [
        '{"real_time":1759843429183,"data":{"car_id":"htcaid_car_trip_1","route_length":68,"destination":"htcaid:node;4239743566","route_cost":68.0,"tick":41,"origin":"htcaid:node;4259655865","event_type":"journey_started"},"tick":41,"event_type":"vehicle_flow","simulation_id":"test_simulation"}',
        '{"real_time":1759843429185,"data":{"car_id":"htcaid_car_trip_1","route_length":68,"route_nodes":"node1,node2,node3","route_links":"link1,link2","destination":"htcaid:node;4239743566","route_cost":68.0,"tick":41,"origin":"htcaid:node;4259655865","event_type":"route_planned"},"tick":41,"event_type":"route_planning","simulation_id":"test_simulation"}',
        '{"real_time":1759843429185,"data":{"cars_in_link":0,"travel_time":1.3438925517165883,"car_id":"htcaid_car_trip_1","event_type":"enter_link","link_id":"htcaid_link_2184","link_capacity":600.0,"link_length":14.932139463517647,"calculated_speed":11.11111111111111,"free_speed":11.11111111111111,"lanes":1,"tick":41},"tick":41,"event_type":"vehicle_flow","simulation_id":"test_simulation"}',
        '{"real_time":1759843429185,"data":{"car_id":"htcaid_car_trip_1","tick":43,"total_distance":14.932139463517647,"event_type":"leave_link","link_id":"htcaid_link_2184","link_length":14.932139463517647},"tick":43,"event_type":"vehicle_flow","simulation_id":"test_simulation"}'
    ]
    
    try:
        with open(test_jsonl_path, 'w') as f:
            for line in test_data:
                f.write(line + '\n')
        
        # Testar extrator
        extractor = HTCJsonlExtractor(test_jsonl_path)
        df, stats = extractor.extract_events()
        
        print("✅ Teste de extração concluído!")
        print(f"📊 Eventos extraídos: {len(df)}")
        print(f"📊 Estatísticas: {stats}")
        
        if not df.empty:
            print("\n📋 Primeiros eventos:")
            print(df.head())
        
        # Limpar arquivo de teste
        Path(test_jsonl_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste simulado: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_comparison_summary(comparison_result):
    """Imprime resumo da comparação"""
    
    print("\n" + "="*80)
    print("📊 RESUMO DA COMPARAÇÃO COM DADOS JSONL")
    print("="*80)
    
    if 'htc_stats' in comparison_result:
        htc_stats = comparison_result['htc_stats']
        print(f"\n📈 Estatísticas HTC (JSONL):")
        print(f"   🚗 Veículos únicos: {htc_stats.get('unique_vehicles', 0):,}")
        print(f"   📊 Total de eventos: {htc_stats.get('total_events', 0):,}")
        print(f"   🗺️ Rotas planejadas: {htc_stats.get('total_routes', 0):,}")
        print(f"   🔗 Links únicos: {htc_stats.get('unique_links', 0):,}")
        print(f"   ⏱️ Duração: {htc_stats.get('simulation_duration', 0):,} ticks")
    
    print(f"\n📊 Contadores de eventos:")
    print(f"   📈 HTC: {comparison_result.get('htc_events_count', 0):,}")
    print(f"   📈 Referência: {comparison_result.get('ref_events_count', 0):,}")
    
    if 'routes_analysis' in comparison_result:
        routes = comparison_result['routes_analysis']
        print(f"\n🗺️ Análise de rotas:")
        print(f"   📊 Rotas HTC: {routes.get('htc_routes_count', 0):,}")
        print(f"   📊 Rotas Ref: {routes.get('ref_routes_count', 0):,}")
        print(f"   🔗 Veículos correspondentes: {routes.get('matching_vehicles', 0):,}")
        print(f"   ℹ️  Explicação: Compara rotas do mesmo veículo entre simuladores")
        
        if 'avg_length_difference' in routes:
            avg_len = routes['avg_length_difference']
            max_len = routes.get('max_length_difference', 0)
            assessment = "🟢 Excelente" if avg_len < 0.05 else "🟡 Aceitável" if avg_len < 0.15 else "🔴 Significativa"
            print(f"   📏 Diferença de comprimento: média {avg_len:.2%}, máx {max_len:.2%} {assessment}")
            
        if 'avg_cost_difference' in routes:
            avg_cost = routes['avg_cost_difference']
            max_cost = routes.get('max_cost_difference', 0)
            assessment = "🟢 Excelente" if avg_cost < 0.05 else "🟡 Aceitável" if avg_cost < 0.15 else "🔴 Significativa"
            print(f"   💰 Diferença de custo: média {avg_cost:.2%}, máx {max_cost:.2%} {assessment}")
            
        if 'avg_complexity_difference' in routes:
            avg_comp = routes['avg_complexity_difference']
            max_comp = routes.get('max_complexity_difference', 0)
            assessment = "🟢 Excelente" if avg_comp < 0.1 else "🟡 Aceitável" if avg_comp < 0.25 else "🔴 Significativa"
            print(f"   🔗 Diferença de complexidade: média {avg_comp:.2%}, máx {max_comp:.2%} {assessment}")
            
        print(f"   💡 Interpretação:")
        print(f"      • Média: tendência geral das diferenças")
        print(f"      • Máx: pior caso encontrado")
        print(f"      • < 5% = Excelente, 5-15% = Aceitável, > 15% = Significativa")
    
    if 'events_analysis' in comparison_result:
        events = comparison_result['events_analysis']
        print(f"\n📊 Análise de eventos:")
        print(f"   📈 Total HTC: {events.get('htc_total', 0):,}")
        print(f"   📈 Total Ref: {events.get('ref_total', 0):,}")
        print(f"   ❌ Tipos ausentes no HTC: {len(events.get('missing_in_htc', []))}")
        print(f"   ❌ Tipos ausentes na Ref: {len(events.get('missing_in_ref', []))}")
    
    if 'temporal_analysis' in comparison_result:
        temporal = comparison_result['temporal_analysis']
        print(f"\n⏱️ Análise temporal:")
        print(f"   📏 Duração HTC: {temporal.get('htc_duration', 0):,} ticks")
        print(f"   📏 Duração Ref: {temporal.get('ref_duration', 0):,} ticks")
        print(f"   📊 Eventos/tick HTC: {temporal.get('htc_events_per_tick', 0):.2f}")
        print(f"   📊 Eventos/tick Ref: {temporal.get('ref_events_per_tick', 0):.2f}")
    
    if 'standard_comparison' in comparison_result:
        std = comparison_result['standard_comparison']
        print(f"\n🔍 Comparação padrão:")
        print(f"   📊 Score de similaridade: {std.similarity_score:.3f}")
        print(f"   🎯 Score de reprodutibilidade: {std.reproducibility_score:.3f}")

if __name__ == "__main__":
    success = test_jsonl_analysis()
    
    if success:
        print("\n✅ Teste concluído com sucesso!")
    else:
        print("\n❌ Teste falhou!")
        sys.exit(1)