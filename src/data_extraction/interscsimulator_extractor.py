"""
Extrator de dados XML para simulações Interscsimulator
"""

import xml.etree.ElementTree as ET
import logging
import glob
import os
from typing import List, Dict, Optional, Iterator
import pandas as pd
from pathlib import Path
import sys

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.models import InterscsimulatorEvent, VehicleEvent

# Tentar importar configurações
try:
    from config.settings import INTERSCSIMULATOR_CONFIG
except ImportError:
    # Configuração padrão se não encontrar o arquivo
    INTERSCSIMULATOR_CONFIG = {
        'data_dir': './data/interscsimulator',
        'file_pattern': '*.xml'
    }


class InterscsimulatorDataExtractor:
    """Extrator de dados das simulações Interscsimulator (XML)"""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(INTERSCSIMULATOR_CONFIG['data_dir'])
        self.file_pattern = INTERSCSIMULATOR_CONFIG['file_pattern']
        self.logger = logging.getLogger(__name__)
        
        # Criar diretório se não existir
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def find_simulation_files(self) -> List[Path]:
        """Encontra todos os arquivos XML de simulação"""
        pattern = str(self.data_dir / self.file_pattern)
        files = [Path(f) for f in glob.glob(pattern)]
        self.logger.info(f"Encontrados {len(files)} arquivos XML em {self.data_dir}")
        return files
    
    def get_simulation_ids(self) -> List[str]:
        """Retorna lista de IDs baseados nos nomes dos arquivos"""
        files = self.find_simulation_files()
        simulation_ids = [f.stem for f in files]  # Nome do arquivo sem extensão
        return simulation_ids
    
    def parse_xml_file(self, file_path: Path) -> List[InterscsimulatorEvent]:
        """Faz parse de um arquivo XML e retorna lista de eventos"""
        events = []
        
        try:
            # Primeiro tentar parsing streaming (mais robusto)
            events = self._parse_xml_streaming(file_path)
            if events:
                self.logger.info(f"Extraídos {len(events)} eventos de {file_path}")
                return events
                
        except ET.ParseError as e:
            self.logger.warning(f"Parsing streaming falhou em {file_path}: {e}")
        except Exception as e:
            self.logger.warning(f"Erro no parsing streaming de {file_path}: {e}")
        
        try:
            # Fallback para parsing com root
            events = self._parse_xml_with_root(file_path)
            if events:
                self.logger.info(f"Extraídos {len(events)} eventos de {file_path}")
                return events
                
        except ET.ParseError as e:
            self.logger.warning(f"Parsing com root falhou em {file_path}: {e}")
        except Exception as e:
            self.logger.warning(f"Erro no parsing com root de {file_path}: {e}")
        
        try:
            # Último recurso: parsing linha por linha
            events = self._parse_xml_line_by_line(file_path)
            
        except Exception as e:
            self.logger.error(f"Erro ao processar arquivo {file_path}: {e}")
        
        self.logger.info(f"Extraídos {len(events)} eventos de {file_path}")
        return events
    
    def _parse_xml_with_root(self, file_path: Path) -> List[InterscsimulatorEvent]:
        """Parse XML com elemento raiz"""
        events = []
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Buscar elementos 'event'
        for event_elem in root.findall('.//event'):
            event = InterscsimulatorEvent.from_xml_element(event_elem)
            events.append(event)
        
        return events
    
    def _parse_xml_streaming(self, file_path: Path) -> List[InterscsimulatorEvent]:
        """Parse XML streaming para arquivos grandes"""
        events = []
        
        # Usar iterparse para memória eficiente
        for event_type, elem in ET.iterparse(file_path, events=('start', 'end')):
            if event_type == 'end' and elem.tag == 'event':
                event = InterscsimulatorEvent.from_xml_element(elem)
                events.append(event)
                elem.clear()  # Liberar memória
        
        return events
    
    def _parse_xml_line_by_line(self, file_path: Path) -> List[InterscsimulatorEvent]:
        """Parse linha por linha para arquivos XML malformados"""
        events = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line.startswith('<event') and line.endswith('/>'):
                    try:
                        # Tentar fazer parse de uma linha de evento
                        elem = ET.fromstring(line)
                        event = InterscsimulatorEvent.from_xml_element(elem)
                        events.append(event)
                    except ET.ParseError as e:
                        self.logger.warning(f"Erro na linha {line_num} de {file_path}: {e}")
                        continue
        
        return events
    
    def get_events_by_simulation(self, file_path: str) -> List[InterscsimulatorEvent]:
        """Retorna eventos de um arquivo XML específico"""
        file_path = Path(file_path)
        
        # Se é apenas um nome, buscar no diretório de dados
        if not file_path.is_absolute() and not file_path.exists():
            file_path = self.data_dir / file_path
            # Se não tem extensão, adicionar .xml
            if not file_path.suffix:
                file_path = file_path.with_suffix('.xml')
        
        if not file_path.exists():
            self.logger.error(f"Arquivo não encontrado: {file_path}")
            return []
        
        return self.parse_xml_file(file_path)
    
    def get_events_by_type(self, file_path: str, event_type: str) -> List[InterscsimulatorEvent]:
        """Retorna eventos de um tipo específico"""
        all_events = self.get_events_by_simulation(file_path)
        filtered_events = [event for event in all_events if event.event_type == event_type]
        self.logger.info(f"Filtrados {len(filtered_events)} eventos do tipo '{event_type}'")
        return filtered_events
    
    def get_vehicle_journey(self, file_path: str, car_id: str) -> List[InterscsimulatorEvent]:
        """Retorna todos os eventos de um veículo específico"""
        all_events = self.get_events_by_simulation(file_path)
        vehicle_events = [event for event in all_events if event.car_id == car_id]
        # Ordenar por timestamp
        vehicle_events.sort(key=lambda x: x.timestamp)
        self.logger.info(f"Encontrados {len(vehicle_events)} eventos para o veículo {car_id}")
        return vehicle_events
    
    def get_simulation_summary(self, file_path: str) -> Dict:
        """Retorna resumo de uma simulação"""
        events = self.get_events_by_simulation(file_path)
        
        if not events:
            return {}
        
        # Extrair ID da simulação do nome do arquivo
        simulation_id = Path(file_path).stem
        
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
            'file_path': str(file_path),
            'total_vehicles': len(vehicle_ids),
            'total_events': len(events),
            'start_time': min(timestamps) if timestamps else 0,
            'end_time': max(timestamps) if timestamps else 0,
            'duration': max(timestamps) - min(timestamps) if timestamps else 0,
            'events_by_type': events_by_type,
            'unique_vehicles': list(vehicle_ids)
        }
        
        return summary
    
    def export_to_dataframe(self, file_path: str) -> pd.DataFrame:
        """Exporta eventos para DataFrame pandas"""
        events = self.get_events_by_simulation(file_path)
        
        # Extrair ID da simulação do nome do arquivo
        simulation_id = Path(file_path).stem
        
        # Converter eventos para dicionários
        data = []
        for event in events:
            row = {
                'simulation_id': simulation_id,
                'timestamp': event.timestamp,
                'car_id': event.car_id,
                'event_type': event.event_type,
                'tick': event.tick
            }
            
            # Adicionar atributos específicos
            for key, value in event.attributes.items():
                if key not in ['time', 'car_id', 'type', 'tick']:
                    row[f"attr_{key}"] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        self.logger.info(f"DataFrame criado com {len(df)} linhas para simulação {simulation_id}")
        return df
    
    def get_multiple_simulations(self, file_paths: List[str]) -> Dict[str, List[InterscsimulatorEvent]]:
        """Extrai dados de múltiplos arquivos"""
        results = {}
        
        for file_path in file_paths:
            sim_id = Path(file_path).stem
            self.logger.info(f"Extraindo dados da simulação: {sim_id} ({file_path})")
            results[sim_id] = self.get_events_by_simulation(file_path)
        
        return results
    
    def get_link_density_data(self, file_path: str) -> pd.DataFrame:
        """Extrai dados de densidade de links do Interscsimulator"""
        enter_events = self.get_events_by_type(file_path, 'enter_link')
        
        density_data = []
        for event in enter_events:
            attrs = event.attributes
            density_data.append({
                'timestamp': event.timestamp,
                'link_id': attrs.get('link_id', ''),
                'cars_in_link': float(attrs.get('cars_in_link', 0)),
                'link_capacity': float(attrs.get('link_capacity', 0)),
                'density': float(attrs.get('cars_in_link', 0)) / max(float(attrs.get('link_capacity', 1)), 1),
                'calculated_speed': float(attrs.get('calculated_speed', 0)),
                'free_speed': float(attrs.get('free_speed', 0)),
                'lanes': float(attrs.get('lanes', 1))
            })
        
        return pd.DataFrame(density_data)
    
    def validate_xml_structure(self, file_path: Path) -> Dict[str, any]:
        """Valida estrutura do arquivo XML"""
        validation_result = {
            'is_valid': False,
            'has_root': False,
            'event_count': 0,
            'event_types': set(),
            'errors': []
        }
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            validation_result['has_root'] = True
            
            events = root.findall('.//event')
            validation_result['event_count'] = len(events)
            
            for event in events[:100]:  # Validar apenas os primeiros 100 eventos
                event_type = event.get('type', '')
                if event_type:
                    validation_result['event_types'].add(event_type)
            
            validation_result['is_valid'] = True
            validation_result['event_types'] = list(validation_result['event_types'])
            
        except ET.ParseError as e:
            validation_result['errors'].append(f"Parse error: {e}")
        except Exception as e:
            validation_result['errors'].append(f"General error: {e}")
        
        return validation_result
    
    def create_sample_xml(self, output_path: Path, num_events: int = 100):
        """Cria arquivo XML de exemplo para testes"""
        root = ET.Element("simulation")
        
        for i in range(num_events):
            event_attrs = {
                'time': str(i * 10),
                'type': 'enter_link' if i % 2 == 0 else 'leave_link',
                'car_id': f'car_{i % 10}',
                'link_id': f'link_{i % 5}',
                'tick': str(i * 10)
            }
            
            if event_attrs['type'] == 'enter_link':
                event_attrs.update({
                    'link_length': '100.0',
                    'link_capacity': '1000.0',
                    'cars_in_link': str(i % 10),
                    'free_speed': '15.0',
                    'calculated_speed': '12.0',
                    'travel_time': '8.33',
                    'lanes': '2'
                })
            else:
                event_attrs.update({
                    'link_length': '100.0',
                    'total_distance': str((i + 1) * 100)
                })
            
            ET.SubElement(root, "event", event_attrs)
        
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        self.logger.info(f"Arquivo XML de exemplo criado: {output_path}")


# Exemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    extractor = InterscsimulatorDataExtractor()
    
    # Criar arquivo de exemplo se não houver dados
    sample_file = extractor.data_dir / "sample_simulation.xml"
    if not sample_file.exists():
        extractor.create_sample_xml(sample_file, 1000)
    
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