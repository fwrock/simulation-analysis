"""
Extrator de dados do Cassandra para simulações HTC
"""

import json
import logging
from typing import List, Dict, Optional, Iterator
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.models import HTCEvent, JourneyStartedEvent, EnterLinkEvent, LeaveLinkEvent, JourneyCompletedEvent

# Tentar importar configurações
try:
    from config.settings import CASSANDRA_CONFIG
except ImportError:
    # Configuração padrão se não encontrar o arquivo
    CASSANDRA_CONFIG = {
        'hosts': ['127.0.0.1'],
        'port': 9042,
        'keyspace': 'htc_reports',
        'table': 'simulation_reports'
    }


class HTCDataExtractor:
    """Extrator de dados das simulações HTC do Cassandra"""
    
    def __init__(self, hosts=None, port=None, keyspace=None):
        self.hosts = hosts or CASSANDRA_CONFIG['hosts']
        self.port = port or CASSANDRA_CONFIG['port']
        self.keyspace = keyspace or CASSANDRA_CONFIG['keyspace']
        self.table = CASSANDRA_CONFIG['table']
        self.cluster = None
        self.session = None
        
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Conecta ao cluster Cassandra"""
        try:
            self.cluster = Cluster(self.hosts, port=self.port)
            self.session = self.cluster.connect()
            self.session.set_keyspace(self.keyspace)
            self.logger.info(f"Conectado ao Cassandra: {self.hosts}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao conectar ao Cassandra: {e}")
            return False
    
    def disconnect(self):
        """Desconecta do cluster"""
        if self.cluster:
            self.cluster.shutdown()
            self.logger.info("Desconectado do Cassandra")
    
    def get_simulation_ids(self) -> List[str]:
        """Retorna lista de IDs de simulação disponíveis"""
        query = "SELECT simulation_id FROM {}".format(self.table)
        try:
            rows = self.session.execute(query)
            # Usar set para eliminar duplicados
            simulation_ids = list(set(row.simulation_id for row in rows))
            self.logger.info(f"Encontrados {len(simulation_ids)} IDs de simulação únicos")
            return simulation_ids
        except Exception as e:
            self.logger.error(f"Erro ao buscar IDs de simulação: {e}")
            return []
    
    def get_events_by_simulation(self, simulation_id: str, limit: Optional[int] = None) -> List[HTCEvent]:
        """Retorna eventos de uma simulação específica"""
        # Usar string formatting direto (seguro em Cassandra)
        if limit:
            query = "SELECT * FROM {} WHERE simulation_id = '{}' LIMIT {} ALLOW FILTERING".format(self.table, simulation_id, limit)
        else:
            query = "SELECT * FROM {} WHERE simulation_id = '{}' ALLOW FILTERING".format(self.table, simulation_id)
        
        try:
            rows = self.session.execute(query)
            events = []
            row_count = 0
            
            for row in rows:
                row_count += 1
                try:
                    # Log da estrutura da primeira linha para debug
                    if row_count == 1:
                        self.logger.debug(f"Primeira linha - Tipo: {type(row)}")
                        self.logger.debug(f"Campos: {row._fields}")
                    
                    event = HTCEvent.from_cassandra_row(row)
                    events.append(event)
                    
                except Exception as row_error:
                    self.logger.error(f"Erro ao processar linha {row_count}: {row_error}")
                    self.logger.debug(f"Dados da linha problemática: {row}")
                    # Tentar acessar campos específicos
                    try:
                        self.logger.debug(f"simulation_id: {getattr(row, 'simulation_id', 'N/A')}")
                        self.logger.debug(f"data: {getattr(row, 'data', 'N/A')}")
                        self.logger.debug(f"created_at: {getattr(row, 'created_at', 'N/A')}")
                    except Exception as field_error:
                        self.logger.debug(f"Erro ao acessar campos: {field_error}")
                    continue
            
            self.logger.info(f"Processadas {row_count} linhas, extraídos {len(events)} eventos da simulação {simulation_id}")
            return events
        except Exception as e:
            self.logger.error(f"Erro ao extrair eventos da simulação {simulation_id}: {e}")
            import traceback
            self.logger.debug(f"Traceback completo: {traceback.format_exc()}")
            return []
    
    def get_events_by_type(self, simulation_id: str, event_type: str) -> List[HTCEvent]:
        """Retorna eventos de um tipo específico"""
        all_events = self.get_events_by_simulation(simulation_id)
        filtered_events = [event for event in all_events if event.event_type == event_type]
        self.logger.info(f"Filtrados {len(filtered_events)} eventos do tipo '{event_type}'")
        return filtered_events
    
    def get_vehicle_journey(self, simulation_id: str, car_id: str) -> List[HTCEvent]:
        """Retorna todos os eventos de um veículo específico"""
        all_events = self.get_events_by_simulation(simulation_id)
        vehicle_events = [event for event in all_events if event.car_id == car_id]
        # Ordenar por timestamp
        vehicle_events.sort(key=lambda x: x.timestamp)
        self.logger.info(f"Encontrados {len(vehicle_events)} eventos para o veículo {car_id}")
        return vehicle_events
    
    def get_typed_events(self, simulation_id: str) -> Dict[str, List]:
        """Retorna eventos organizados por tipo com objetos específicos"""
        all_events = self.get_events_by_simulation(simulation_id)
        typed_events = {
            'journey_started': [],
            'enter_link': [],
            'leave_link': [],
            'journey_completed': []
        }
        
        for event in all_events:
            event_type = event.event_type
            
            if event_type == 'journey_started':
                typed_events['journey_started'].append(JourneyStartedEvent.from_htc_event(event))
            elif event_type == 'enter_link':
                typed_events['enter_link'].append(EnterLinkEvent.from_htc_event(event))
            elif event_type == 'leave_link':
                typed_events['leave_link'].append(LeaveLinkEvent.from_htc_event(event))
            elif event_type == 'journey_completed':
                typed_events['journey_completed'].append(JourneyCompletedEvent.from_htc_event(event))
        
        return typed_events
    
    def get_simulation_summary(self, simulation_id: str) -> Dict:
        """Retorna resumo de uma simulação"""
        events = self.get_events_by_simulation(simulation_id)
        
        if not events:
            return {}
        
        # Contar eventos por tipo
        events_by_type = {}
        vehicle_ids = set()
        timestamps = []
        
        for event in events:
            event_type = event.event_type
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
            vehicle_ids.add(event.car_id)
            timestamps.append(event.timestamp)
        
        summary = {
            'simulation_id': simulation_id,
            'total_vehicles': len(vehicle_ids),
            'total_events': len(events),
            'start_time': min(timestamps) if timestamps else 0,
            'end_time': max(timestamps) if timestamps else 0,
            'duration': max(timestamps) - min(timestamps) if timestamps else 0,
            'events_by_type': events_by_type,
            'unique_vehicles': list(vehicle_ids)
        }
        
        return summary
    
    def export_to_dataframe(self, simulation_id: str) -> pd.DataFrame:
        """Exporta eventos para DataFrame pandas"""
        events = self.get_events_by_simulation(simulation_id)
        
        # Converter eventos para dicionários
        data = []
        for event in events:
            row = {
                'simulation_id': event.simulation_id,
                'timestamp': event.timestamp,
                'car_id': event.car_id,
                'event_type': event.event_type,
                'tick': event.tick,
                'node_id': event.node_id,
                'report_type': event.report_type,
                'created_at': event.created_at
            }
            
            # Adicionar dados específicos do evento
            for key, value in event.data.items():
                row[f"data_{key}"] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        self.logger.info(f"DataFrame criado com {len(df)} linhas para simulação {simulation_id}")
        return df
    
    def get_multiple_simulations(self, simulation_ids: List[str]) -> Dict[str, List[HTCEvent]]:
        """Extrai dados de múltiplas simulações"""
        results = {}
        
        for sim_id in simulation_ids:
            self.logger.info(f"Extraindo dados da simulação: {sim_id}")
            results[sim_id] = self.get_events_by_simulation(sim_id)
        
        return results
    
    def get_link_density_data(self, simulation_id: str) -> pd.DataFrame:
        """Extrai dados de densidade de links"""
        enter_events = self.get_events_by_type(simulation_id, 'enter_link')
        
        density_data = []
        for event in enter_events:
            if hasattr(event, 'data'):
                data = event.data
                density_data.append({
                    'timestamp': event.timestamp,
                    'link_id': data.get('link_id', ''),
                    'cars_in_link': data.get('cars_in_link', 0),
                    'link_capacity': data.get('link_capacity', 0),
                    'density': data.get('cars_in_link', 0) / max(data.get('link_capacity', 1), 1),
                    'calculated_speed': data.get('calculated_speed', 0),
                    'free_speed': data.get('free_speed', 0),
                    'lanes': data.get('lanes', 1)
                })
        
        return pd.DataFrame(density_data)
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# Exemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    with HTCDataExtractor() as extractor:
        # Listar simulações disponíveis
        sim_ids = extractor.get_simulation_ids()
        print(f"Simulações disponíveis: {sim_ids}")
        
        if sim_ids:
            # Analisar primeira simulação
            sim_id = sim_ids[0]
            summary = extractor.get_simulation_summary(sim_id)
            print(f"Resumo da simulação {sim_id}: {summary}")
            
            # Exportar para DataFrame
            df = extractor.export_to_dataframe(sim_id)
            print(f"DataFrame shape: {df.shape}")
            print(df.head())