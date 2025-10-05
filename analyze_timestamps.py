#!/usr/bin/env python3
"""
Script para analisar timestamps dos dados de simulação
"""

import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

try:
    from config.settings import CASSANDRA_CONFIG, INTERSCSIMULATOR_CONFIG
except ImportError:
    CASSANDRA_CONFIG = {
        'hosts': ['127.0.0.1'],
        'port': 9042,
        'keyspace': 'htc_reports',
        'table': 'simulation_reports'
    }
    INTERSCSIMULATOR_CONFIG = {
        'data_dir': 'data/interscsimulator',
        'file_pattern': '*.xml'
    }

from src.data_extraction.htc_extractor import HTCDataExtractor
from src.data_extraction.interscsimulator_extractor import InterscsimulatorDataExtractor
import pandas as pd
import numpy as np

def analyze_timestamps():
    print("=== Análise de Timestamps ===")
    
    # Analisar HTC
    print("\n--- HTC Data ---")
    htc_extractor = HTCDataExtractor()
    if htc_extractor.connect():
        htc_events = htc_extractor.get_events_by_simulation("bfs_cenario_1000_viagens_2", limit=1000)
        htc_extractor.disconnect()
        
        if htc_events:
            htc_timestamps = [event.timestamp for event in htc_events[:20]]
            print(f"Primeiros 20 timestamps HTC: {htc_timestamps}")
            print(f"Min timestamp: {min(htc_timestamps)}, Max: {max(htc_timestamps)}")
            print(f"Tipo de dados: {type(htc_events[0].timestamp)}")
            
            # Verificar se são sequenciais
            sorted_timestamps = sorted(htc_timestamps)
            print(f"Sequencial? {htc_timestamps == sorted_timestamps}")
            
            # Análise de gaps
            diffs = np.diff(sorted_timestamps)
            print(f"Diferenças entre timestamps: min={min(diffs)}, max={max(diffs)}, avg={np.mean(diffs)}")
    else:
        print("Não foi possível conectar ao Cassandra")
    
    # Analisar Interscsimulator  
    print("\n--- Interscsimulator Data ---")
    interscs_extractor = InterscsimulatorDataExtractor()
    interscs_events = interscs_extractor.get_events_by_simulation("/home/my_user/interscsimulator/output/base_scenario_1000_n/events.xml")
    
    if interscs_events:
        # Pegar uma amostra menor para análise
        interscs_timestamps = [event.timestamp for event in interscs_events[:1000:50]]  # A cada 50 eventos
        print(f"Amostra de 20 timestamps Interscsimulator: {interscs_timestamps}")
        print(f"Min timestamp: {min(interscs_timestamps)}, Max: {max(interscs_timestamps)}")
        print(f"Tipo de dados: {type(interscs_events[0].timestamp)}")
        
        # Verificar se são sequenciais
        sorted_timestamps = sorted(interscs_timestamps)
        print(f"Sequencial? {interscs_timestamps == sorted_timestamps}")
        
        # Análise de gaps
        diffs = np.diff(sorted_timestamps)
        print(f"Diferenças entre timestamps: min={min(diffs)}, max={max(diffs)}, avg={np.mean(diffs)}")
        
        # Analisar distribuição temporal
        print(f"\nDistribuição temporal (primeiros 1000 eventos):")
        df = pd.DataFrame({'timestamp': [e.timestamp for e in interscs_events[:1000]]})
        print(df['timestamp'].describe())
    
    # Comparar escalas temporais
    print("\n=== Comparação de Escalas ===")
    if htc_events and interscs_events:
        htc_range = max([e.timestamp for e in htc_events]) - min([e.timestamp for e in htc_events])
        interscs_range = max([e.timestamp for e in interscs_events[:1000]]) - min([e.timestamp for e in interscs_events[:1000]])
        
        print(f"Range temporal HTC: {htc_range}")
        print(f"Range temporal Interscsimulator (amostra): {interscs_range}")
        print(f"Razão (HTC/Interscsimulator): {htc_range / interscs_range if interscs_range > 0 else 'N/A'}")

if __name__ == "__main__":
    analyze_timestamps()