"""
Extrator de dados dos arquivos JSONL do HTC (Hyperbolic Time Chamber)
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class HTCJsonlExtractor:
    """Extrator para arquivos JSONL do HTC"""
    
    def __init__(self, jsonl_file_path: str):
        self.jsonl_file_path = Path(jsonl_file_path)
        self.events = []
        self.routes_data = []
        
    def extract_events(self, batch_size: int = 50000) -> Tuple[pd.DataFrame, Dict]:
        """
        Extrai eventos do arquivo JSONL processando em lotes
        
        Args:
            batch_size: Tamanho do lote para processamento em memória
            
        Returns:
            Tuple com DataFrame dos eventos e dicionário de estatísticas
        """
        
        if not self.jsonl_file_path.exists():
            logger.error(f"Arquivo JSONL não encontrado: {self.jsonl_file_path}")
            return pd.DataFrame(), {}
            
        logger.info(f"🔄 Extraindo eventos de {self.jsonl_file_path}")
        
        all_events = []
        routes = {}
        event_counts = defaultdict(int)
        total_lines = 0
        
        try:
            with open(self.jsonl_file_path, 'r', encoding='utf-8') as f:
                batch_events = []
                
                for line_num, line in enumerate(f, 1):
                    if line_num % 10000 == 0:
                        logger.info(f"   📊 Processadas {line_num:,} linhas...")
                    
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        event_data = json.loads(line)
                        
                        # Extrair dados básicos do evento
                        event = {
                            'real_time': event_data.get('real_time'),
                            'tick': event_data.get('tick'),
                            'event_type': event_data.get('event_type'),
                            'simulation_id': event_data.get('simulation_id')
                        }
                        
                        # Extrair dados específicos do evento
                        data = event_data.get('data', {})
                        event_type = data.get('event_type')
                        
                        if event_type:
                            event['data_event_type'] = event_type
                            event_counts[event_type] += 1
                            
                            # Extrair dados específicos por tipo de evento
                            if event_type == 'journey_started':
                                event.update({
                                    'car_id': data.get('car_id'),
                                    'origin': data.get('origin'),
                                    'destination': data.get('destination'),
                                    'route_length': data.get('route_length'),
                                    'route_cost': data.get('route_cost')
                                })
                                
                            elif event_type == 'route_planned':
                                car_id = data.get('car_id')
                                route_links = data.get('route_links', '').split(',') if data.get('route_links') else []
                                route_nodes = data.get('route_nodes', '').split(',') if data.get('route_nodes') else []
                                
                                routes[car_id] = {
                                    'origin': data.get('origin'),
                                    'destination': data.get('destination'),
                                    'route_links': route_links,
                                    'route_nodes': route_nodes,
                                    'route_length': data.get('route_length'),
                                    'route_cost': data.get('route_cost')
                                }
                                event.update({
                                    'car_id': car_id,
                                    'origin': data.get('origin'),
                                    'destination': data.get('destination'),
                                    'route_length': data.get('route_length'),
                                    'route_cost': data.get('route_cost'),
                                    'route_links_count': len(route_links),
                                    'route_nodes_count': len(route_nodes)
                                })
                                
                            elif event_type in ['enter_link', 'leave_link']:
                                event.update({
                                    'car_id': data.get('car_id'),
                                    'link_id': data.get('link_id'),
                                    'link_length': data.get('link_length'),
                                    'link_capacity': data.get('link_capacity'),
                                    'cars_in_link': data.get('cars_in_link'),
                                    'free_speed': data.get('free_speed'),
                                    'calculated_speed': data.get('calculated_speed'),
                                    'travel_time': data.get('travel_time'),
                                    'lanes': data.get('lanes'),
                                    'total_distance': data.get('total_distance')
                                })
                        
                        batch_events.append(event)
                        
                        # Processar em lotes
                        if len(batch_events) >= batch_size:
                            all_events.extend(batch_events)
                            batch_events = []
                            logger.info(f"   💾 Processados {len(all_events):,} eventos até agora...")
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Erro ao decodificar JSON na linha {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Erro ao processar linha {line_num}: {e}")
                        continue
                
                # Adicionar último lote
                if batch_events:
                    all_events.extend(batch_events)
            
            logger.info(f"✅ Processadas {total_lines:,} linhas, {len(all_events):,} eventos extraídos")
            
        except Exception as e:
            logger.error(f"Erro ao ler arquivo JSONL: {e}")
            return pd.DataFrame(), {}
        
        # Converter para DataFrame
        df_events = pd.DataFrame(all_events)
        
        # Estatísticas
        stats = {
            'total_events': len(all_events),
            'events_by_type': dict(event_counts),
            'unique_vehicles': len(set(event.get('car_id') for event in all_events if event.get('car_id'))),
            'total_routes': len(routes),
            'simulation_duration': df_events['tick'].max() if not df_events.empty else 0,
            'unique_links': len(set(event.get('link_id') for event in all_events if event.get('link_id'))),
            'routes_data': routes
        }
        
        logger.info(f"📊 Estatísticas extraídas:")
        logger.info(f"   📈 Total de eventos: {stats['total_events']:,}")
        logger.info(f"   🚗 Veículos únicos: {stats['unique_vehicles']:,}")
        logger.info(f"   🗺️ Rotas planejadas: {stats['total_routes']:,}")
        logger.info(f"   🔗 Links únicos: {stats['unique_links']:,}")
        logger.info(f"   ⏱️ Duração da simulação: {stats['simulation_duration']:,} ticks")
        
        return df_events, stats
    
    def extract_route_analysis(self, routes_data: Dict) -> pd.DataFrame:
        """
        Extrai análise de rotas
        
        Args:
            routes_data: Dicionário com dados das rotas
            
        Returns:
            DataFrame com análise das rotas
        """
        
        route_analysis = []
        
        for car_id, route_info in routes_data.items():
            analysis = {
                'car_id': car_id,
                'origin': route_info.get('origin'),
                'destination': route_info.get('destination'),
                'route_length': route_info.get('route_length'),
                'route_cost': route_info.get('route_cost'),
                'links_count': len(route_info.get('route_links', [])),
                'nodes_count': len(route_info.get('route_nodes', [])),
                'links_per_km': len(route_info.get('route_links', [])) / route_info.get('route_length', 1) if route_info.get('route_length', 0) > 0 else 0,
                'cost_per_km': route_info.get('route_cost', 0) / route_info.get('route_length', 1) if route_info.get('route_length', 0) > 0 else 0
            }
            route_analysis.append(analysis)
        
        return pd.DataFrame(route_analysis)
    
    def compare_routes_with_reference(self, htc_routes: Dict, ref_routes: Dict) -> Dict:
        """
        Compara rotas do HTC com rotas de referência
        
        Args:
            htc_routes: Rotas do HTC
            ref_routes: Rotas de referência
            
        Returns:
            Dicionário com comparação das rotas
        """
        
        comparison = {
            'matching_origin_destination': 0,
            'different_route_length': 0,
            'different_route_cost': 0,
            'different_links_count': 0,
            'route_efficiency_comparison': [],
            'detailed_comparisons': []
        }
        
        for car_id, htc_route in htc_routes.items():
            # Tentar encontrar rota correspondente na referência
            ref_route = ref_routes.get(car_id)
            
            if ref_route:
                # Verificar origem e destino
                same_od = (htc_route.get('origin') == ref_route.get('origin') and 
                          htc_route.get('destination') == ref_route.get('destination'))
                
                if same_od:
                    comparison['matching_origin_destination'] += 1
                    
                    # Comparar características da rota
                    htc_length = htc_route.get('route_length', 0)
                    ref_length = ref_route.get('route_length', 0)
                    
                    htc_cost = htc_route.get('route_cost', 0)
                    ref_cost = ref_route.get('route_cost', 0)
                    
                    htc_links = len(htc_route.get('route_links', []))
                    ref_links = len(ref_route.get('route_links', []))
                    
                    if abs(htc_length - ref_length) > 0.1:  # Tolerância de 0.1
                        comparison['different_route_length'] += 1
                    
                    if abs(htc_cost - ref_cost) > 0.1:
                        comparison['different_route_cost'] += 1
                    
                    if htc_links != ref_links:
                        comparison['different_links_count'] += 1
                    
                    # Análise de eficiência
                    efficiency = {
                        'car_id': car_id,
                        'htc_length': htc_length,
                        'ref_length': ref_length,
                        'length_diff': htc_length - ref_length,
                        'htc_cost': htc_cost,
                        'ref_cost': ref_cost,
                        'cost_diff': htc_cost - ref_cost,
                        'htc_links': htc_links,
                        'ref_links': ref_links,
                        'links_diff': htc_links - ref_links
                    }
                    
                    comparison['route_efficiency_comparison'].append(efficiency)
                    comparison['detailed_comparisons'].append({
                        'car_id': car_id,
                        'htc_route': htc_route,
                        'ref_route': ref_route,
                        'same_origin_destination': same_od
                    })
        
        return comparison