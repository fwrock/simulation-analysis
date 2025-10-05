#!/usr/bin/env python3
"""
Script para testar consultas Cassandra
"""

from cassandra.cluster import Cluster
import json

def test_cassandra():
    cluster = Cluster(['127.0.0.1'], port=9042)
    session = cluster.connect()
    session.set_keyspace('htc_reports')
    
    # Testar query simples
    print("=== Testando query básica ===")
    try:
        result = session.execute("SELECT * FROM simulation_reports LIMIT 1")
        for row in result:
            print(f"Estrutura da row: {type(row)}")
            print(f"Campos disponíveis: {dir(row)}")
            if hasattr(row, '_fields'):
                print(f"Campos da tabela: {row._fields}")
            break
    except Exception as e:
        print(f"Erro na query básica: {e}")
    
    # Testar query com WHERE
    print("\n=== Testando query com WHERE ===")
    simulation_id = "bfs_cenario_1000_viagens_2"
    
    # Versão com format
    try:
        query1 = "SELECT * FROM simulation_reports WHERE simulation_id = ? ALLOW FILTERING"
        result = session.execute(query1, [simulation_id])
        rows = list(result)
        print(f"Query com '?' retornou {len(rows)} linhas")
    except Exception as e:
        print(f"Erro query com '?': {e}")
    
    # Versão com string direto
    try:
        query2 = f"SELECT * FROM simulation_reports WHERE simulation_id = '{simulation_id}' ALLOW FILTERING"
        result = session.execute(query2)
        rows = list(result)
        print(f"Query com string direto retornou {len(rows)} linhas")
    except Exception as e:
        print(f"Erro query string direto: {e}")
    
    # Listar simulation_ids disponíveis
    print("\n=== IDs de simulação disponíveis ===")
    try:
        result = session.execute("SELECT DISTINCT simulation_id FROM simulation_reports")
        sim_ids = [row.simulation_id for row in result]
        print(f"IDs encontrados: {sim_ids[:10]}")  # Primeiros 10
    except Exception as e:
        print(f"Erro ao listar IDs: {e}")
    
    cluster.shutdown()

if __name__ == "__main__":
    test_cassandra()