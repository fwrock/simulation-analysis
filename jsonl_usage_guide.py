#!/usr/bin/env python3
"""
Exemplo de uso do sistema de análise com dados JSONL
"""

import sys
from pathlib import Path

def create_example_usage():
    """Cria exemplos de uso do sistema"""
    
    print("🚀 Sistema de Análise de Simulações - Exemplos de Uso")
    print("="*80)
    
    print("\n📊 1. ANÁLISE COM DADOS JSONL:")
    print("   Compara HTC (via arquivo JSONL) com Interscsimulator (via XML)")
    print("   Comando:")
    print("   python main.py --mode jsonl \\")
    print("                  --htc-jsonl /path/to/htc/events.jsonl \\")
    print("                  --ref-sim /path/to/interscs/events.xml \\")
    print("                  --output-dir ./output/comparison")
    
    print("\n📊 2. ANÁLISE PADRÃO (CASSANDRA + XML):")
    print("   Compara HTC (via Cassandra) com Interscsimulator (via XML)")
    print("   Comando:")
    print("   python main.py --mode compare \\")
    print("                  --htc-sim bfs_cenario_1000_viagens_2 \\")
    print("                  --ref-sim data/interscs/events.xml \\")
    print("                  --output-dir ./output/comparison")
    
    print("\n📊 3. ANÁLISE INDIVIDUAL:")
    print("   Analisa uma simulação individualmente")
    print("   Comando:")
    print("   python main.py --mode single \\")
    print("                  --htc-sim bfs_cenario_1000_viagens_2 \\")
    print("                  --simulator htc \\")
    print("                  --output-dir ./output/single")
    
    print("\n📊 4. ANÁLISE DE VEÍCULO ESPECÍFICO:")
    print("   Compara jornada de um veículo específico")
    print("   Comando:")
    print("   python main.py --mode vehicle \\")
    print("                  --htc-sim bfs_cenario_1000_viagens_2 \\")
    print("                  --ref-sim data/interscs/events.xml \\")
    print("                  --vehicle-id trip_1 \\")
    print("                  --output-dir ./output/vehicle")
    
    print("\n📊 5. ANÁLISE EM LOTE:")
    print("   Analisa múltiplas simulações")
    print("   Comando:")
    print("   python main.py --mode batch \\")
    print("                  --htc-sims sim1 sim2 sim3 \\")
    print("                  --ref-sims events1.xml events2.xml events3.xml \\")
    print("                  --output-dir ./output/batch")
    
    print("\n" + "="*80)
    print("🔧 ESTRUTURA DOS DADOS JSONL:")
    print("="*80)
    
    print("\n📁 Localização esperada dos arquivos JSONL do HTC:")
    print("   /home/dean/hyperbolic-time-chamber/output/reports/json/{simulationId}/*.jsonl")
    
    print("\n📋 Formato dos eventos JSONL:")
    print('   {"real_time":1759843429183,"data":{"car_id":"htcaid_car_trip_1",...},"tick":41,"event_type":"vehicle_flow",...}')
    
    print("\n📊 Tipos de eventos suportados:")
    print("   - journey_started: Início de viagem")
    print("   - route_planned: Planejamento de rota")
    print("   - enter_link: Entrada em link")
    print("   - leave_link: Saída de link")
    
    print("\n" + "="*80)
    print("📊 FUNCIONALIDADES DA ANÁLISE JSONL:")
    print("="*80)
    
    print("\n🗺️ Análise de Rotas:")
    print("   - Comparação de comprimento de rotas")
    print("   - Comparação de custo de rotas")
    print("   - Análise de complexidade (número de links)")
    print("   - Identificação de rotas divergentes")
    
    print("\n📊 Análise de Eventos:")
    print("   - Distribuição de tipos de eventos")
    print("   - Contagem total de eventos")
    print("   - Identificação de eventos ausentes")
    print("   - Análise de proporções")
    
    print("\n⏱️ Análise Temporal:")
    print("   - Duração das simulações")
    print("   - Taxa de eventos por tick")
    print("   - Picos de atividade")
    print("   - Comparação de temporização")
    
    print("\n📈 Visualizações Geradas:")
    print("   - Gráficos de distribuição de eventos")
    print("   - Análise de densidade de links")
    print("   - Métricas temporais")
    print("   - Dashboard interativo")
    print("   - Relatórios HTML detalhados")
    
    print("\n" + "="*80)
    print("🛠️ CONFIGURAÇÃO DO AMBIENTE:")
    print("="*80)
    
    print("\n1. Ativar ambiente virtual:")
    print("   source venv/bin/activate")
    
    print("\n2. Instalar dependências (se necessário):")
    print("   pip install -r requirements.txt")
    
    print("\n3. Verificar estrutura de arquivos:")
    print("   - HTC JSONL: /home/dean/hyperbolic-time-chamber/output/reports/json/")
    print("   - Interscs XML: data/interscsimulator/")
    print("   - Saída: output/")
    
    print("\n" + "="*80)
    print("📋 EXEMPLO PRÁTICO:")
    print("="*80)
    
    example_commands = [
        "# Ativar ambiente",
        "source venv/bin/activate",
        "",
        "# Análise JSONL para cenário 2500 veículos", 
        "python main.py --mode jsonl \\",
        "               --htc-jsonl /home/dean/hyperbolic-time-chamber/output/reports/json/bfs_cenario_2500_viagens_3/events.jsonl \\",
        "               --ref-sim data/interscsimulator/events_bfs_cenario_2500_viagens_3.xml \\",
        "               --output-dir output/2500",
        "",
        "# Resultados em:",
        "# - output/2500/comparison_jsonl_*.json",
        "# - output/2500/comparison_report.html", 
        "# - output/2500/jsonl_analysis_report.html",
        "# - output/2500/plots/*.png"
    ]
    
    for cmd in example_commands:
        print(f"   {cmd}")

def test_with_available_data():
    """Testa com dados disponíveis no workspace"""
    
    print("\n" + "="*80)
    print("🧪 TESTE COM DADOS DISPONÍVEIS:")
    print("="*80)
    
    # Verificar dados disponíveis
    available_comparisons = list(Path("output").glob("*/comparison_*.json"))
    
    if available_comparisons:
        print(f"\n📁 Encontrados {len(available_comparisons)} arquivos de comparação:")
        for comp in available_comparisons[:5]:  # Mostrar apenas os primeiros 5
            print(f"   - {comp}")
        
        print(f"\n💡 Você pode usar os dados XML correspondentes para teste:")
        print("   python main.py --mode jsonl \\")
        print("                  --htc-jsonl /path/to/your/events.jsonl \\")
        print("                  --ref-sim data/interscsimulator/events.xml")
    else:
        print("\n❌ Nenhum arquivo de comparação encontrado em output/")
        print("💡 Execute primeiro uma análise padrão para gerar dados de referência")
    
    # Verificar se há dados JSONL disponíveis
    jsonl_base = Path("/home/dean/hyperbolic-time-chamber/output/reports/json")
    if jsonl_base.exists():
        jsonl_files = list(jsonl_base.glob("*/events.jsonl"))
        if jsonl_files:
            print(f"\n📁 Encontrados {len(jsonl_files)} arquivos JSONL:")
            for jsonl in jsonl_files[:3]:  # Mostrar apenas os primeiros 3
                print(f"   - {jsonl}")
            
            print(f"\n✅ Você pode testar com:")
            print(f"   python main.py --mode jsonl \\")
            print(f"                  --htc-jsonl {jsonl_files[0]} \\")
            print(f"                  --ref-sim data/interscsimulator/events.xml")
        else:
            print(f"\n❌ Nenhum arquivo JSONL encontrado em {jsonl_base}")
    else:
        print(f"\n❌ Diretório {jsonl_base} não encontrado")

if __name__ == "__main__":
    create_example_usage()
    test_with_available_data()
    
    print("\n" + "="*80)
    print("✅ PRONTO PARA USO!")
    print("="*80)
    print("\n💡 Execute um dos comandos acima para começar a análise.")
    print("📧 Para dúvidas, consulte a documentação ou os logs gerados.")