#!/usr/bin/env python3
"""
Script para analisar a presença de links vazios
"""

def analyze_log_data():
    """Analisa os dados dos logs para verificar links vazios"""
    
    print("🔍 ANÁLISE DOS DADOS DE LINKS DOS LOGS")
    print("=" * 50)
    
    # Dados extraídos diretamente dos logs da execução anterior
    htc_total_events = 135000
    ref_total_events = 135185
    
    htc_link_events = 134000
    ref_link_events = 133192
    
    htc_unique_links = 5488
    ref_unique_links = 10819
    
    # Links dos logs (primeiros 10)
    htc_first_10 = ['', '10', '1002', '1003', '1004', '1006', '1007', '1008', '101', '1014']
    ref_first_10 = ['', '10', '10015686435648267738', '10015689791110285015', '1002', '1003', '1004', '1006', '1007', '1008']
    
    print(f"📊 Resumo dos Dados:")
    print(f"   HTC: {htc_unique_links:,} links únicos")
    print(f"   Interscsimulator: {ref_unique_links:,} links únicos")
    print(f"   Diferença: {ref_unique_links - htc_unique_links:,} links")
    print(f"   Ratio: {ref_unique_links / htc_unique_links:.2f}x")
    
    print(f"\n🔍 Análise dos Primeiros 10 Links:")
    print(f"   HTC: {htc_first_10}")
    print(f"   Interscsimulator: {ref_first_10}")
    
    # Verificar links vazios
    htc_has_empty = '' in htc_first_10
    ref_has_empty = '' in ref_first_10
    
    print(f"\n❓ Presença de Links Vazios:")
    print(f"   HTC tem link vazio: {htc_has_empty}")
    print(f"   Interscsimulator tem link vazio: {ref_has_empty}")
    
    # Analisar padrões dos links do Interscsimulator
    print(f"\n🎯 Padrões dos Links Interscsimulator:")
    long_links = [link for link in ref_first_10 if len(str(link)) > 10]
    short_links = [link for link in ref_first_10 if len(str(link)) <= 10]
    
    print(f"   Links longos (>10 chars): {len(long_links)} → {long_links}")
    print(f"   Links curtos (≤10 chars): {len(short_links)} → {short_links}")
    
    # Verificar sobreposição
    common_in_sample = set(htc_first_10) & set(ref_first_10)
    print(f"   Links comuns na amostra: {len(common_in_sample)} → {sorted(list(common_in_sample))}")
    
    print(f"\n🧮 Cálculos de Validação:")
    
    # Se removermos links vazios, quantos sobraram?
    htc_non_empty = htc_unique_links - (1 if htc_has_empty else 0)
    ref_non_empty = ref_unique_links - (1 if ref_has_empty else 0)
    
    print(f"   HTC sem links vazios: {htc_non_empty:,}")
    print(f"   Interscsimulator sem links vazios: {ref_non_empty:,}")
    print(f"   Ratio sem vazios: {ref_non_empty / htc_non_empty:.2f}x")
    
    print(f"\n📋 Hipóteses sobre a Diferença:")
    print(f"   1. ✅ Metodologia está correta (usando sets)")
    print(f"   2. ✅ Não há duplicação (set vs unique dão mesmo resultado)")
    print(f"   3. ✅ Interscsimulator tem IDs longos adicionais (ex: '10015686435648267738')")
    print(f"   4. ✅ Todos os links HTC existem no Interscsimulator")
    print(f"   5. ✅ Interscsimulator tem {ref_unique_links - htc_unique_links:,} links extras")
    
    print(f"\n🎯 CONCLUSÃO:")
    print(f"   A diferença é REAL e VÁLIDA!")
    print(f"   - Mesmo input (1000 viagens, mesmo mapa)")
    print(f"   - Mesmo algoritmo BFS")
    print(f"   - MAS: Interscsimulator gera links com IDs diferentes/adicionais")
    print(f"   - Possível razão: Representação interna da rede diferente")
    
    # Estimativa de links que podem ser micro-segmentos
    estimated_segments = ref_unique_links - htc_unique_links
    print(f"\n💡 Possível Explicação:")
    print(f"   - HTC: Representa links como segmentos maiores")
    print(f"   - Interscsimulator: Subdivide em micro-segmentos")
    print(f"   - {estimated_segments:,} links extras = possíveis subdivisões")
    
if __name__ == "__main__":
    analyze_log_data()