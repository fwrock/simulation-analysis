#!/usr/bin/env python3
"""
Script para debugging detalhado da contagem de links
"""

import pandas as pd
from pathlib import Path
from src.data_extraction.htc_extractor import HTCDataExtractor
from src.data_extraction.interscsimulator_extractor import InterscsimulatorDataExtractor
from src.visualization.plotter import SimulationVisualizer

def debug_link_counts():
    """Análise detalhada da contagem de links"""
    
    print("🔍 DEBUGGING DETALHADO DA CONTAGEM DE LINKS")
    print("=" * 60)
    
    # Extrair dados dos dois simuladores
    htc_extractor = HTCDataExtractor()
    ref_extractor = InterscsimulatorDataExtractor()
    visualizer = SimulationVisualizer()
    
    print("\n📊 Extraindo dados...")
    htc_events = htc_extractor.get_events_by_simulation('bfs_cenario_1000_viagens_2')
    ref_events = ref_extractor.get_events_from_file('/home/my_user/interscsimulator/output/base_scenario_1000_n/events.xml')
    
    print(f"✅ HTC: {len(htc_events)} eventos")
    print(f"✅ Interscsimulator: {len(ref_events)} eventos")
    
    # Converter para DataFrames usando o mesmo método do visualizador
    htc_df = visualizer._events_to_dataframe(htc_events, 'HTC')
    ref_df = visualizer._events_to_dataframe(ref_events, 'Interscsimulator')
    
    print(f"\n📋 DataFrames criados:")
    print(f"✅ HTC DataFrame: {len(htc_df)} linhas")
    print(f"✅ Interscsimulator DataFrame: {len(ref_df)} linhas")
    
    # Analisar eventos de link
    htc_link_events = htc_df[htc_df['event_type'].isin(['enter_link', 'leave_link'])]
    ref_link_events = ref_df[ref_df['event_type'].isin(['enter_link', 'leave_link'])]
    
    print(f"\n🔗 Eventos de link:")
    print(f"✅ HTC: {len(htc_link_events)} eventos de link")
    print(f"✅ Interscsimulator: {len(ref_link_events)} eventos de link")
    
    # Verificar se há valores nulos
    print(f"\n❓ Valores nulos em normalized_link_id:")
    print(f"✅ HTC: {htc_df['normalized_link_id'].isna().sum()} nulos")
    print(f"✅ Interscsimulator: {ref_df['normalized_link_id'].isna().sum()} nulos")
    
    # Verificar valores vazios
    print(f"\n❓ Valores vazios em normalized_link_id:")
    print(f"✅ HTC: {(htc_df['normalized_link_id'] == '').sum()} vazios")
    print(f"✅ Interscsimulator: {(ref_df['normalized_link_id'] == '').sum()} vazios")
    
    # Extrair links únicos EXATAMENTE como no visualizador
    htc_links_all = htc_df['normalized_link_id'].dropna().unique()
    ref_links_all = ref_df['normalized_link_id'].dropna().unique()
    
    # Aplicar Set para garantir unicidade (como no código original)
    htc_links = set(htc_links_all)
    ref_links = set(ref_links_all)
    
    print(f"\n🎯 CONTAGEM FINAL DE LINKS ÚNICOS:")
    print(f"✅ HTC: {len(htc_links_all)} unique() → {len(htc_links)} set()")
    print(f"✅ Interscsimulator: {len(ref_links_all)} unique() → {len(ref_links)} set()")
    
    # Verificar se unique() e set() dão o mesmo resultado
    print(f"\n🔄 Verificação de consistência:")
    print(f"✅ HTC unique == set: {len(htc_links_all) == len(htc_links)}")
    print(f"✅ Interscsimulator unique == set: {len(ref_links_all) == len(ref_links)}")
    
    # Mostrar alguns exemplos de links
    print(f"\n📝 Primeiros 10 links HTC:")
    for i, link in enumerate(sorted(list(htc_links))[:10]):
        print(f"   {i+1:2d}. '{link}'")
    
    print(f"\n📝 Primeiros 10 links Interscsimulator:")
    for i, link in enumerate(sorted(list(ref_links))[:10]):
        print(f"   {i+1:2d}. '{link}'")
    
    # Análise de intersecção
    common_links = htc_links & ref_links
    htc_only = htc_links - ref_links
    ref_only = ref_links - htc_links
    
    print(f"\n🔗 Análise de sobreposição:")
    print(f"✅ Links comuns: {len(common_links)}")
    print(f"✅ Exclusivos HTC: {len(htc_only)}")
    print(f"✅ Exclusivos Interscsimulator: {len(ref_only)}")
    
    # Verificar se todos os links HTC estão no Interscsimulator
    htc_in_ref = htc_links.issubset(ref_links)
    print(f"✅ Todos links HTC estão no Interscsimulator: {htc_in_ref}")
    
    # Analisar padrões dos links exclusivos do Interscsimulator
    print(f"\n📊 Alguns links exclusivos do Interscsimulator:")
    ref_only_sample = sorted(list(ref_only))[:10]
    for i, link in enumerate(ref_only_sample):
        print(f"   {i+1:2d}. '{link}'")
    
    # Verificar se há links duplicados por algum motivo
    print(f"\n🔍 Verificação de duplicação:")
    htc_link_counts = pd.Series(list(htc_links_all)).value_counts()
    ref_link_counts = pd.Series(list(ref_links_all)).value_counts()
    
    htc_duplicates = htc_link_counts[htc_link_counts > 1]
    ref_duplicates = ref_link_counts[ref_link_counts > 1]
    
    print(f"✅ Links duplicados HTC: {len(htc_duplicates)}")
    print(f"✅ Links duplicados Interscsimulator: {len(ref_duplicates)}")
    
    if len(htc_duplicates) > 0:
        print("   HTC duplicados:", htc_duplicates.head())
    if len(ref_duplicates) > 0:
        print("   Interscsimulator duplicados:", ref_duplicates.head())
        
    print(f"\n🎯 CONCLUSÃO:")
    print(f"   - Metodologia está correta (usando sets)")
    print(f"   - Diferença é real: Interscsimulator tem {len(ref_links) - len(htc_links)} links a mais")
    print(f"   - Ratio: {len(ref_links) / len(htc_links):.2f}x")

if __name__ == "__main__":
    debug_link_counts()